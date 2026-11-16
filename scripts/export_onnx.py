"""Export a Conv-TasNet checkpoint to the block-streaming ONNX model used by deploy/realtime_test.py.

  python scripts/export_onnx.py --ckpt f12.pt --out-dir deploy/

Writes <out-dir>/f12_stream_block512.onnx (name configurable) and model_meta.json. The graph is StreamCore(model,
block) wrapped by FlatCore, opset 17: one call consumes one (2, block) block; every state tensor is an explicit
input with a matching *_out output (all states start at zero; feed outputs[1:] back as inputs[1:]).
Verification: block-by-block ONNX Runtime streaming vs whole-signal offline PyTorch, max |err| < 1e-4, on --wav if
given, otherwise on 30 s of noise."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf
import torch

from rtvs.checks import random_input
from rtvs.common import SR
from rtvs.inference import export_convtasnet_onnx, ort_session
from rtvs.model import CausalConvTasNet
from rtvs.stream import run_stream


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="deploy")
    ap.add_argument("--name", default="f12_stream_block512.onnx")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--wav", default=None, help="44.1 kHz audio for the verification (first 30 s)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = CausalConvTasNet(**c["model_cfg"]); m.load_state_dict(c["model"]); m.eval()
    path = out / a.name
    _, names, init, core = export_convtasnet_onnx(m, a.block, str(path))
    sess = ort_session(str(path), 1)

    def step(b, st):
        r = sess.run(None, dict(zip(names, [b.numpy()] + st))); return torch.from_numpy(r[0]), r[1:]

    if a.wav:
        xa, sr = sf.read(a.wav, dtype="float32", always_2d=True, frames=30 * SR)
        assert sr == SR, f"{a.wav}: {sr} Hz, need {SR}"
        x = torch.from_numpy((xa if xa.shape[1] == 2 else xa[:, :1].repeat(2, 1)).T.copy())
        audio = os.path.basename(a.wav)
    else:
        x = random_input(30.0)[0]; audio = "30 s of Gaussian noise (std 0.1)"
    with torch.inference_mode():
        ref = m(x[None])[0]
        est, _ = run_stream(step, [s.copy() for s in init], x, a.block, left=core.left)
    err = float((ref - est).abs().max()); assert err < 1e-4, err
    meta = {"model": f"causal Conv-TasNet L={m.L} S={m.S} (stereo in, stereo vocals out)", "epoch": c.get("epoch"),
            "checkpoint_sha256": hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest(),
            "onnx_file": path.name, "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "onnx_opset": 17,
            "sample_rate": SR, "block": a.block, "window_L": m.L, "hop_S": m.S, "output_delay_samples": core.left,
            "output_delay_note": f"output block c = vocals estimate for input samples [c*{a.block} - delay, c*{a.block} + {a.block} - delay)",
            "algorithmic_latency_ms": (m.L - 1) / SR * 1e3,
            "inputs": [{"name": i.name, "shape": i.shape} for i in sess.get_inputs()],
            "outputs": [{"name": o.name, "shape": o.shape} for o in sess.get_outputs()],
            "state_rule": "all states start at zeros; feed outputs[1:] back as inputs[1:] in the same order",
            "verification": {"what": "block-by-block ORT streaming vs whole-signal offline PyTorch", "audio": audio,
                             "max_abs_err": err, "threshold": 1e-4, "pass": err < 1e-4}}
    with open(out / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ["onnx_file", "output_delay_samples", "algorithmic_latency_ms", "verification"]}),
          path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
