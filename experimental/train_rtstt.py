"""Train the RT-STT adaptation (four-source causal TFC-TDF network; comparison experiment). Needs a CUDA GPU.

  python -m experimental.train_rtstt --out runs/rtstt [--config experimental/configs/rtstt_train.json]

Settings are in the JSON config (AdamW 1e-4, effective batch 32 via gradient accumulation, L1 over the four stems,
64 chunks per train track per epoch, plateau / early stop on the 3-epoch moving average of the valid vocals uSDR
median, best.pt = max raw valid vocals uSDR median). Validation runs causal stateful inference on the 14 whole
valid tracks. Resumable at epoch boundaries: re-run the same command."""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from experimental.rtstt_chunked import infer_track
from experimental.rtstt_paper import RTSTT
from rtvs.audio import load_track
from rtvs.common import add_data_args, apply_data_args, dump_json, provenance, read_split
from rtvs.data import MusdbChunksMulti
from rtvs.metrics import usdr

SOURCES = ["vocals", "drums", "bass", "other"]


def tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on; torch.backends.cudnn.allow_tf32 = on


def save(path, m, opt, state, c):
    tmp = str(path) + ".tmp"
    torch.save({"model": m.state_dict(), "model_cfg": m.cfg, "optimizer": opt.state_dict(), "state": state, "config": c,
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "numpy": np.random.get_state(), "python": random.getstate()}}, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def valid(m):
    m.eval(); tf32(False); rows = []
    for name in read_split("valid"):
        data, _ = load_track(name, "valid", dtype="float32")
        est = infer_track(m, data["mixture"], "cuda", 32768)
        rows.append({"track": name, **{s: usdr(data[s], est[i]) for i, s in enumerate(SOURCES)}})
    scores = {s: {"mean": float(np.mean([r[s] for r in rows])), "median": float(np.median([r[s] for r in rows]))}
              for s in SOURCES}
    m.train(); tf32(True)
    return scores, rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=str(Path(__file__).with_name("configs") / "rtstt_train.json"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--time_budget_min", type=float, default=None, help="stop at a clean epoch boundary (exit 4)")
    add_data_args(p)
    a = p.parse_args(); apply_data_args(a)
    assert torch.cuda.is_available(), "training needs a CUDA GPU"
    torch.set_num_threads(1)
    c = json.load(open(a.config)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
    m = RTSTT().cuda(); opt = torch.optim.AdamW(m.parameters(), lr=c["lr"], weight_decay=c["weight_decay"]); tf32(True)
    state = {"epoch": 0, "best_epoch": None, "best_valid_vocals_usdr": -1e9, "best_ma": -1e9, "since_best": 0,
             "since_cut": 0, "stopped": None, "history": []}
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        assert ck["config"] == c and ck["model_cfg"] == m.cfg, "config or architecture changed on resume"
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); state = ck["state"]
        torch.set_rng_state(ck["rng"]["torch"]); torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        np.random.set_state(ck["rng"]["numpy"]); random.setstate(ck["rng"]["python"])
        if str(state.get("stopped", "")).startswith("time budget"):
            state["stopped"] = None
        assert not state["stopped"], "a convergence / safety stop is final"
    dump_json(c, str(out / "config.json"))
    train = read_split("train")
    n = len(train) * c["chunks_per_track_per_epoch"]
    ds = MusdbChunksMulti(train, "train", c["chunk_samples"] / 44100, items=n * c["max_epochs"], mix_prob=.5, seed=c["seed"])
    start = time.time(); m.train()
    for ep in range(state["epoch"] + 1, c["max_epochs"] + 1):
        recent = state["history"][-1]["epoch_wall_seconds"] / 60 if state["history"] else 5
        if a.time_budget_min is not None and (time.time() - start) / 60 + recent + 2 > a.time_budget_min:
            state["stopped"] = "time budget at clean epoch boundary"; save(last, m, opt, state, c); break
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range((ep - 1) * n, ep * n)),
                                             batch_size=c["micro_batch"], shuffle=False, num_workers=a.workers,
                                             drop_last=True, pin_memory=True)
        opt.zero_grad(); losses = []; t = time.time(); torch.cuda.reset_peak_memory_stats()
        for i, (mix, stems) in enumerate(loader):
            loss = (m(mix.cuda()) - stems.cuda()).abs().mean(); assert torch.isfinite(loss), "non-finite training loss"
            (loss / c["gradient_accumulation"]).backward(); losses.append(float(loss))
            if (i + 1) % c["gradient_accumulation"] == 0:
                gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), c["gradient_clip_l2"]))
                assert math.isfinite(gn), "non-finite gradient"
                opt.step(); opt.zero_grad()
        train_s = time.time() - t; score, tracks = valid(m); v = score["vocals"]["median"]
        assert math.isfinite(v), "non-finite valid score"
        history = state["history"]
        ma = float(np.mean([h["valid_vocals_usdr_median"] for h in history[-2:]] + [v]))
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "valid_vocals_usdr_median": v, "valid_usdr_sources": score,
               "per_track": tracks, "valid_vocals_usdr_ma": ma, "train_seconds": train_s,
               "epoch_wall_seconds": time.time() - t, "lr": opt.param_groups[0]["lr"],
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        history.append(row); state["epoch"] = ep
        if v > state["best_valid_vocals_usdr"]:
            state.update(best_valid_vocals_usdr=v, best_epoch=ep)
            tmp = out / "best.pt.tmp"
            torch.save({"model": m.state_dict(), "model_cfg": m.cfg, "epoch": ep, "valid": row, "config": c}, tmp)
            os.replace(tmp, out / "best.pt")
        if ma > state["best_ma"]:
            state.update(best_ma=ma, since_best=0, since_cut=0)
        else:
            state["since_best"] += 1; state["since_cut"] += 1
            if state["since_cut"] >= c["plateau_patience"]:
                for group in opt.param_groups:
                    group["lr"] *= c["lr_factor"]
                state["since_cut"] = 0
            if state["since_best"] >= c["early_stop_patience"]:
                state["stopped"] = "early stop: valid vocals moving average plateau"
        if ep == c["max_epochs"]:
            state["stopped"] = state["stopped"] or "max epochs reached"
        save(last, m, opt, state, c)
        dump_json({"label": "valid only", "config": c, "state": state, "provenance": provenance()}, str(out / "train_log.json"))
        print(json.dumps({k: v for k, v in row.items() if k != "per_track"}), flush=True)
        if state["stopped"]:
            break
    dump_json({"label": "valid only", "config": c, "state": state, "provenance": provenance()}, str(out / "train_log.json"))
    if str(state["stopped"]).startswith("time budget"):
        sys.exit(4)


if __name__ == "__main__":
    main()
