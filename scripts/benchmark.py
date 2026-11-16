"""Single-thread CPU timing of block-512 ONNX Runtime streaming (software timing; no audio device involved).

  python scripts/benchmark.py --out timing/ F23=f23.pt F12=f12.pt F6=f6.pt

Protocol of the README table: ONNX Runtime CPU EP, 1 intra-op / 1 inter-op thread, sequential; block 512
(budget 512 / 44100 = 11.61 ms); >= 60 s of audio per run after a 2 s warm-up, first 20 calls discarded;
3 repetitions per model, interleaved; process pinned to one core (Linux). Before timing, streaming output is
checked against the offline output (< 1e-4). RTF = median call time / block duration.
The README numbers were measured on an otherwise idle (exclusively allocated) Intel Xeon Gold 6150 node."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import soundfile as sf
import torch

from rtvs.checks import random_input
from rtvs.common import SR, add_data_args, apply_data_args, dump_json, provenance, read_split, track_dir
from rtvs.inference import load_checkpoint, make_stream
from rtvs.stream import run_stream

BLOCK = 512
DROP = 20


def cpu_name():
    try:
        for s in Path("/proc/cpuinfo").read_text().splitlines():
            if s.startswith("model name"):
                return s.split(":", 1)[1].strip()
    except Exception:
        pass
    import platform
    return platform.processor() or platform.machine()


def load_audio(a):
    if a.wav:
        x, sr = sf.read(a.wav, dtype="float32", always_2d=True, frames=int(a.seconds * SR))
        assert sr == SR
        return torch.from_numpy(x[:, :2].T.copy()), os.path.basename(a.wav)
    if a.musdb_root or os.environ.get("MUSDB18HQ_ROOT"):
        name = read_split("valid")[0]
        x, _ = sf.read(f"{track_dir(name, 'valid')}/mixture.wav", dtype="float32", always_2d=True,
                       start=30 * SR, frames=int(a.seconds * SR))
        return torch.from_numpy(x.T.copy()), f"valid track 0 from 30 s ({name})"
    return random_input(a.seconds)[0], "Gaussian noise"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="LABEL=checkpoint.pt (architecture inferred from the checkpoint)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--wav", default=None, help="audio to stream (default: MUSDB valid track 0 if available, else noise)")
    add_data_args(ap)
    a = ap.parse_args(); apply_data_args(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
    x, audio = load_audio(a)
    budget = BLOCK / SR * 1e3
    models = {}
    for spec in a.models:
        label, ck = spec.split("=", 1)
        m, arch = load_checkpoint(ck)
        step, init = make_stream(m, arch, BLOCK, "onnxruntime", 1)
        with torch.inference_mode():
            ref = m(x[None])[0]
            est, _ = run_stream(step, init(), x, BLOCK, left=m.left)
        err = float((ref - est).abs().max()); assert err < 1e-4, (label, err)
        models[label] = (m, step, init, hashlib.sha256(Path(ck).read_bytes()).hexdigest(), err)
    rows = []
    for rep in range(1, a.reps + 1):
        for label, (m, step, init, sha, err) in models.items():
            with torch.inference_mode():
                run_stream(step, init(), x[:, :2 * SR], BLOCK, left=m.left)          # warm-up
                _, t = run_stream(step, init(), x, BLOCK, left=m.left)
            t = np.array(t[DROP:]) * 1e3
            r = dict(model=label, repetition=rep, block=BLOCK, threads=1, checkpoint_sha256=sha,
                     consistency_max_abs_err=err, audio_seconds=a.seconds, n_calls=int(len(t)), budget_ms=budget,
                     p50_ms=float(np.median(t)), p99_ms=float(np.percentile(t, 99)), max_ms=float(t.max()),
                     mean_ms=float(t.mean()), overrun_ratio=float((t > budget).mean()),
                     rtf_median=float(np.median(t) / budget), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            rows.append(r); print(json.dumps(r), flush=True)
    import onnxruntime as ort
    dump_json({"label": "ORT single-thread block-512 software timing (not hardware end-to-end latency)",
               "cpu_model": cpu_name(), "onnxruntime": ort.__version__, "input": audio, "rows": rows,
               "provenance": provenance()}, str(out / "summary.json"))


if __name__ == "__main__":
    main()
