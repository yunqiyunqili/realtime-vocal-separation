"""Causality probe + streaming consistency + size for a Conv-TasNet checkpoint (or a random-init model).

  python scripts/check_model.py --config F12                 # random init, theory check
  python scripts/check_model.py --ckpt f12.pt --wav song.wav # trained model on real audio

Checks: (1) future-dependency probe on the offline path, (2) offline vs reference per-hop streaming,
(3) offline vs StreamCore block streaming (eager) and (4) vs ONNX Runtime block streaming, all with tolerance 1e-4.
Exit code 1 if any check fails."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf
import torch

from rtvs.checks import block_consistency, future_probe, model_stats, random_input, step_consistency
from rtvs.common import SR, dump_json, provenance
from rtvs.inference import make_stream
from rtvs.model import CausalConvTasNet

TIERS = {"F23": dict(L=1024, S=256), "F12": dict(L=512, S=128), "F6": dict(L=256, S=64)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="Conv-TasNet checkpoint; default: random init of --config")
    ap.add_argument("--config", default="F12", choices=list(TIERS))
    ap.add_argument("--wav", default=None, help="44.1 kHz stereo audio for the checks (default: 6 s of noise)")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(1)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        m = CausalConvTasNet(**ck["model_cfg"]); m.load_state_dict(ck["model"])
    else:
        m = CausalConvTasNet(mask_act="sigmoid", **TIERS[a.config])
    m.eval()
    if a.wav:
        x, sr = sf.read(a.wav, dtype="float32", always_2d=True, frames=int(a.seconds * SR))
        assert sr == SR, f"{a.wav}: {sr} Hz, need {SR}"
        x = torch.from_numpy((x if x.shape[1] == 2 else x[:, :1].repeat(2, 1)).T.copy())[None]
    else:
        x = random_input(a.seconds)
    res = {"model": a.ckpt or f"random init {a.config}", "stats": model_stats(m)}
    res["future_probe"] = future_probe(m, x)
    res["step_consistency"] = step_consistency(m, x)
    for impl in ("eager", "onnxruntime"):
        step, init = make_stream(m, "convtasnet", a.block, impl, 1)
        res[f"block_consistency_{impl}"] = block_consistency(m, step, init, x[0], a.block)
    fp = res["future_probe"]
    ok = {"probe": fp["all_match_theory"] if not a.ckpt else fp["causal_within_window"],
          "step": res["step_consistency"]["pass"], "eager": res["block_consistency_eager"]["pass"],
          "onnx": res["block_consistency_onnxruntime"]["pass"]}
    res["pass"] = ok
    print(json.dumps({"stats": {k: v for k, v in res["stats"].items() if k != "receptive_field"},
                      "lookahead_min_max": [fp["lookahead_min"], fp["lookahead_max"]],
                      "errors": {k: res[k]["max_abs_err"] for k in res if "consistency" in k}, "pass": ok}, indent=1))
    if a.out:
        res["provenance"] = provenance(); dump_json(res, a.out)
    sys.exit(0 if all(ok.values()) else 1)


if __name__ == "__main__":
    main()
