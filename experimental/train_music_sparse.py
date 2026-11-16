"""Train a MusicSparse variant (comparison experiment; see experimental/README.md). Needs a CUDA GPU.

  python -m experimental.train_music_sparse --variant fft512 --out runs/C      # model C (11.6 ms)
  python -m experimental.train_music_sparse --variant asym   --out runs/S      # model S (11.6 ms, asymmetric window)

Recipe (identical for all variants): L1 waveform loss on vocals, Adam 3e-4, batch 1 x 6 s, per epoch 8 chunks from
every train track (no cross-track mixing, fixed per-epoch seeds), grad-clip 5, FP32 (no TF32), 30-epoch cap,
ReduceLROnPlateau(factor 0.5, patience 5) and early stop after 8 epochs without improvement.
Per-epoch selection score = median vocals museval SDR on the 14 valid tracks, input 28-32 s, scored on 30-32 s.
Before training, a random-init causality probe and a streaming-consistency check must pass (probe.json).
Resumable: re-run the same command (state is saved every 100 updates)."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from experimental.asym_music import AsymMusic
from experimental.fast_music import FastMusic
from rtvs import io as rio
from rtvs.common import STEMS, add_data_args, apply_data_args, provenance, read_split, track_dir
from rtvs.data import MusdbChunks
from rtvs.metrics import evaluate_track

VARIANTS = ("fft512", "asym", "stride8", "original")


def write(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(d, indent=2)); tmp.replace(p)


def build(variant):
    return AsymMusic() if variant == "asym" else FastMusic(variant)


@torch.no_grad()
def validate(m):
    m.eval(); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rows = []
    for name in read_split("valid"):
        d = {k: rio.read(f"{track_dir(name, 'valid')}/{k}.wav", start=28 * 44100, frames=4 * 44100, dtype="float32",
                         always_2d=True)[0] for k in ["mixture", *STEMS]}
        pred = m(torch.from_numpy(d["mixture"].T.copy())[None].cuda())[0, :, 88200:].cpu().numpy().T
        ref = d["vocals"][88200:]
        r = evaluate_track(d["mixture"][88200:], ref, (d["drums"] + d["bass"] + d["other"])[88200:], pred)
        r.update(track=name, l1=float(np.mean(np.abs(pred - ref))))
        rows.append(r)
    return {"score": float(np.nanmedian([r["sdr_vocals"] for r in rows])),
            "usdr_median": float(np.median([r["usdr_vocals"] for r in rows])),
            "valid_l1": float(np.mean([r["l1"] for r in rows])), "rows": rows}


@torch.no_grad()
def probe(m, out):
    """Random-init causality probe (no output may change before t0 - (L-1)) and step-vs-forward consistency."""
    m = m.cpu().eval(); torch.set_num_threads(1)
    x = torch.randn(1, 2, 8192) * .05; ref = m(x); rows = []
    for phase in sorted({0, 1, m.H // 4, m.H // 2, m.H - 2, m.H - 1}):
        t = 4096 + phase; z = x.clone(); z[:, :, t:] += torch.randn_like(z[:, :, t:]) * .1
        ids = torch.where((m(z) - ref).abs().amax((0, 1)) > 1e-7)[0]; assert len(ids) > 0, "unresponsive model"
        first = int(ids[0]); rows.append({"t0": t, "first_changed": first, "lookahead": t - first})
        assert first >= t - (m.L - 1), rows
    st = m.init_state(); ys = []; xp = torch.nn.functional.pad(x, (0, m.L))
    for i in range(0, xp.shape[-1], m.H):
        y, st = m.step(xp[:, :, i:i + m.H], st); ys.append(y)
    err = float((torch.cat(ys, -1)[:, :, m.left:m.left + x.shape[-1]] - ref).abs().max()); assert err < 1e-4, err
    write(out / "probe.json", {"passed": True, "rows": rows, "stream_error": err,
                               "params": sum(p.numel() for p in m.parameters()),
                               "dense_gmacs_s": sum(m.dense_macs_per_hop().values()) * 44100 / m.H / 1e9})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--out", required=True)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--time_budget_min", type=float, default=None, help="save and exit 4 after this many minutes")
    add_data_args(p)
    a = p.parse_args(); apply_data_args(a); v = a.variant
    assert torch.cuda.is_available(), "training needs a CUDA GPU"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print("already complete"); return
    torch.manual_seed(42); np.random.seed(42); random.seed(42); torch.set_num_threads(2)
    m = build(v)
    if not (out / "last.pt").exists():
        probe(m, out)
    m = m.cuda(); opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5, patience=5, min_lr=1e-6)
    st = {"epoch": 1, "cursor": 0, "losses": [], "history": [], "best": -1e9, "bad": 0, "updates": 0,
          "audio_hours": 0., "wall_seconds": 0.}
    if (out / "last.pt").exists():
        c = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        m.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"]); sched.load_state_dict(c["scheduler"]); st = c["state"]
        torch.set_rng_state(c["rng"]["torch"]); torch.cuda.set_rng_state_all(c["rng"]["cuda"])
        np.random.set_state(c["rng"]["numpy"]); random.setstate(c["rng"]["python"])
    t0 = time.monotonic(); prior = st["wall_seconds"]

    def save(path):
        st["wall_seconds"] = prior + time.monotonic() - t0
        c = {"model": m.state_dict(), "model_cfg": {"variant": m.variant}, "optimizer": opt.state_dict(),
             "scheduler": sched.state_dict(), "state": st,
             "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                     "numpy": np.random.get_state(), "python": random.getstate()}}
        tmp = Path(str(path) + ".tmp"); torch.save(c, tmp); tmp.replace(path)

    train = read_split("train"); patience = 8
    for epoch in range(st["epoch"], a.max_epochs + 1):
        st["epoch"] = epoch; m.train()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

        def stream():
            for ti, name in enumerate(train):
                ds = MusdbChunks([name], chunk_s=6, items=8, mix_prob=0, seed=42 + 1000 * epoch + ti)
                for j in range(8):
                    if ti * 8 + j >= st["cursor"]:
                        x, y = ds[j]; yield x[None], y[None]
        for x, y in stream():
            if a.time_budget_min is not None and time.monotonic() - t0 > a.time_budget_min * 60:
                save(out / "last.pt"); print("time budget reached; re-run to resume", flush=True); sys.exit(4)
            x = x.cuda(); y = y.cuda(); opt.zero_grad(set_to_none=True)
            loss = (m(x) - y).abs().mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5, error_if_nonfinite=True); assert torch.isfinite(loss)
            opt.step(); st["losses"].append(float(loss.detach())); st["updates"] += 1; st["cursor"] += 1
            st["audio_hours"] += 6 / 3600
            if st["cursor"] % 100 == 0:
                save(out / "last.pt")
        ve = validate(m); score = ve["score"]
        rec = {"epoch": epoch, "train_l1": float(np.mean(st["losses"])), "valid": ve, "updates": st["updates"],
               "audio_hours": st["audio_hours"], "lr": opt.param_groups[0]["lr"]}
        improved = score > st["best"] + 1e-4
        if improved:
            st["best"] = score; st["best_epoch"] = epoch
        sched.step(score); st["bad"] = 0 if improved else st["bad"] + 1
        st["history"].append(rec); st["epoch"] = epoch + 1; st["cursor"] = 0; st["losses"] = []
        save(out / "last.pt")
        if improved:
            save(out / "best.pt")
        write(out / "curves.json", st); print("EPOCH", v, epoch, score, "BEST", st["best"], flush=True)
        if st["bad"] >= patience:
            break
    st["stop_reason"] = "early_stopping" if st["bad"] >= patience else "epoch_cap"
    write(out / "summary.json", {"state": st, "provenance": provenance(),
                                 "best_sha256": hashlib.sha256((out / "best.pt").read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
