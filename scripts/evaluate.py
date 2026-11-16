"""Whole-track streaming evaluation on MUSDB18-HQ (the protocol behind every number in the README).

Inference : ONNX Runtime CPU, 1 thread per worker process, stateful block streaming, block 512, fixed model delay
            removed. No whole-track normalisation, no lookahead, no chunk averaging.
Metrics   : museval BSS Eval v4, 1-s windows, joint [vocals, accompaniment = mixture - vocals estimate];
            vocals SDR = median over tracks of the per-track frame medians.
            Also uSDR (whole track; mean and median over tracks), SI-SDR, accompaniment SDR / uSDR,
            vocals-only museval.
NaN diagnosis per 1-s window (a window is NaN iff any of ref vocals / ref acc / est vocals / est acc is all-zero):
            A = reference vocals silent, B = reference voiced but estimate all-zero, C = other. B windows are
            dropped by museval although the model output silence over singing, which can bias SDR upwards, so
            they are counted separately.
Summaries are written for all 50 test tracks and for 49 tracks without 'PR - Oh No'.

--condition effects applies the fixed random effects chain of rtvs/effects.py (one seed per track) to the stems
and evaluates against the processed vocals (uSDR is the primary metric there: reverb tails make windows that are
silent in the dry reference non-silent, which produces extreme museval frame values).

Test-set rule used for the README: model selection on valid only, each final model evaluated on test once. The
script therefore requires --confirm-test for --split test and refuses to run when summary.json already exists.
Per-track results are written as they finish, so an interrupted run resumes where it stopped."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch

from rtvs.audio import load_track
from rtvs.common import STEMS, add_data_args, dump_json, provenance, read_split, set_musdb_root
from rtvs.inference import ARCHS, load_checkpoint, make_stream
from rtvs.metrics import evaluate_track, summarize
from rtvs.stream import run_stream

BLOCK, WIN = 512, 44100
_W = {}


def _init(ckpt, arch, split, condition, musdb):
    set_musdb_root(musdb)
    torch.set_num_threads(1)
    m, arch = load_checkpoint(ckpt, arch)
    step, init = make_stream(m, arch, BLOCK, "onnxruntime", 1)
    torch.manual_seed(123)
    x = torch.randn(2, 8192) * .05
    with torch.inference_mode():
        ref = m(x[None])[0]
        est, _ = run_stream(step, init(), x, BLOCK, left=m.left)
    err = float((ref - est).abs().max())
    assert err < 1e-4, f"streaming output differs from offline output: {err}"
    _W.update(step=step, init=init, left=m.left, split=split, condition=condition, stream_check=err)


def nan_diag(voc, acc, est_v, est_a):
    n = len(voc) // WIN
    cnt = {"n_windows": n, "nan": 0, "A_ref_vocals_silent": 0, "B_ref_voiced_est_zero": 0, "C_other": 0,
           "ref_voiced_windows": 0}
    for t in range(n):
        sl = slice(t * WIN, (t + 1) * WIN)
        rv, ra = not np.any(voc[sl]), not np.any(acc[sl])
        ev, ea = not np.any(est_v[sl]), not np.any(est_a[sl])
        if not rv:
            cnt["ref_voiced_windows"] += 1
        if rv or ra or ev or ea:
            cnt["nan"] += 1
            if rv:
                cnt["A_ref_vocals_silent"] += 1
            elif ev:
                cnt["B_ref_voiced_est_zero"] += 1
            else:
                cnt["C_other"] += 1
    return cnt


def job(index, name):
    split, cond = _W["split"], _W["condition"]
    w, sr = load_track(name, split, dtype="float32")
    effect = None
    if cond == "effects":
        from rtvs.effects import effect_seed, transform
        s, effect = transform(np.stack([w[k].T for k in STEMS]), effect_seed(split, index))
        mix, voc, acc = s.sum(0).T, s[0].T, s[1:].sum(0).T
    else:
        mix, voc, acc = w["mixture"], w["vocals"], w["drums"] + w["bass"] + w["other"]
    t = time.time()
    with torch.inference_mode():
        est, times = run_stream(_W["step"], _W["init"](), torch.from_numpy(mix.T.copy()), BLOCK, left=_W["left"])
    wall = time.time() - t
    est = est.numpy().T.astype(np.float64)
    mix, voc, acc = mix.astype(np.float64), voc.astype(np.float64), acc.astype(np.float64)
    r = evaluate_track(mix, voc, acc, est)
    r.update(track=name, track_index=index, condition=cond, effect=effect, block=BLOCK, stream_wall_s=wall,
             audio_s=len(est) / sr, block_ms_median=float(np.median(times) * 1e3),
             block_ms_p99=float(np.percentile(times, 99) * 1e3),
             rms_ratio_est_over_ref=float(np.sqrt(np.mean(est ** 2)) / max(np.sqrt(np.mean(voc ** 2)), 1e-12)),
             **{"diag_" + k: v for k, v in nan_diag(voc, acc, est, mix - est).items()})
    return r


def summary(df):
    s = summarize(df.to_dict("records"))
    keys = ["n_windows", "nan", "A_ref_vocals_silent", "B_ref_voiced_est_zero", "C_other", "ref_voiced_windows"]
    tot = {k: int(df["diag_" + k].sum()) for k in keys}
    tot["museval_nan_frames_joint_sum"] = int(df["nan_frames_joint"].sum()) if df["nan_frames_joint"].notna().all() else None
    tot["diag_matches_museval"] = tot["museval_nan_frames_joint_sum"] == tot["nan"]
    s["nan_diagnosis"] = tot
    s["rms_ratio_median"] = float(df.rms_ratio_est_over_ref.median())
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default=None, choices=ARCHS, help="default: inferred from the checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--condition", default="clean", choices=["clean", "effects"])
    ap.add_argument("--workers", type=int, default=8, help="parallel processes (each runs 1 ORT thread)")
    ap.add_argument("--max-tracks", type=int, default=0, help="evaluate only the first N tracks (smoke test)")
    ap.add_argument("--confirm-test", action="store_true",
                    help="required for --split test: test is for the final, valid-selected model only")
    add_data_args(ap)
    a = ap.parse_args()
    if a.split == "test" and not a.confirm_test:
        sys.exit("refusing: --split test needs --confirm-test (select models on valid; evaluate test once)")
    if os.path.exists(os.path.join(a.out, "summary.json")):
        sys.exit("refusing: summary.json exists in --out (completed evaluations are not overwritten)")
    tdir = os.path.join(a.out, "tracks"); os.makedirs(tdir, exist_ok=True)
    tf = lambda i: os.path.join(tdir, f"track_{i:02d}.json")
    names = read_split(a.split)
    if a.max_tracks:
        names = names[:a.max_tracks]
    todo = [i for i in range(len(names)) if not os.path.exists(tf(i))]
    print(f"{len(names)} {a.split} tracks, {len(todo)} to run", flush=True)
    musdb = a.musdb_root or os.environ.get("MUSDB18HQ_ROOT")
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.ckpt, a.arch, a.split, a.condition, musdb)) as ex:
        futs = {ex.submit(job, i, names[i]): i for i in todo}
        for f in as_completed(futs):
            r = f.result(); dump_json(r, tf(futs[f]))
            print(f"{r['track']}: sdr_v={r['sdr_vocals']:.2f} usdr={r['usdr_vocals']:.2f} "
                  f"B={r['diag_B_ref_voiced_est_zero']}", flush=True)
    df = pd.DataFrame([json.load(open(tf(i))) for i in range(len(names))])
    df.drop(columns=["effect"]).to_csv(os.path.join(a.out, "per_track.csv"), index=False)
    out = {"ckpt": os.path.basename(a.ckpt), "split": a.split, "condition": a.condition,
           "inference": "ONNX Runtime CPU, 1 thread, stateful block streaming, block 512, delay-aligned",
           "all": summary(df), "provenance": provenance()}
    if a.split == "test" and len(names) == 50:
        out["without_PR_Oh_No_49"] = summary(df[df.track != "PR - Oh No"])
    dump_json(out, os.path.join(a.out, "summary.json"))
    keys = ["sdr_vocals_median_of_track_medians", "usdr_vocals_mean", "usdr_vocals_median", "nan_diagnosis"]
    print(json.dumps({k: out["all"][k] for k in keys}, indent=1, default=str))


if __name__ == "__main__":
    main()
