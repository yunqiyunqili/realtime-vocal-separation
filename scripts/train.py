"""Train a causal Conv-TasNet vocals model on MUSDB18-HQ (train split; model selection on the 14 valid tracks).

Configs (only the encoder window L / hop S differ between latency tiers):
  F23  L=1024, S=256  (23.2 ms algorithmic latency)
  F12  L=512,  S=128  (11.6 ms)
  F6   L=256,  S=64   (5.8 ms)
  E    F23 window with a ReLU mask and the shorter plateau / early-stop schedule (earlier baseline)
Common recipe: L1 waveform loss, sigmoid mask, Adam lr 3e-4 with linear warm-up over epoch 1, batch 16 random
6-s chunks (64 per train track per epoch), per-stem gain U[0.25, 1.25] and L/R swap, cross-track mixing p=0.5,
grad-clip 5, TF32 for training only. Every epoch: whole-track causal validation on the 14 valid tracks
(uSDR / SI-SDR / RMS ratio). "Improvement" = 3-epoch moving average of the valid uSDR median; plateau 8 epochs
-> lr x0.5, 20 epochs without improvement -> early stop. best.pt = max raw valid uSDR median.
Safety valves stop training on collapse (RMS ratio, dead mask channels, very low uSDR).
Checkpoints in --out: best.pt, last.pt (full state, resumable), snap_epXXX.pt (every 10 epochs), epochs/epXXX.pt.

Resume: run the same command again; training continues from last.pt (config changes are refused).
Exit codes: 3 = safety valve / non-finite loss, 4 = --time_budget_min reached (resumable)."""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from rtvs.audio import load_track
from rtvs.common import add_data_args, apply_data_args, dump_json, provenance, read_split
from rtvs.data import MusdbChunks
from rtvs.metrics import si_sdr, usdr
from rtvs.model import CausalConvTasNet

BASE = dict(loss="l1", mask="relu", lr=3e-4, warmup_epochs=1, batch=16, chunks_per_track=64, max_epochs=100,
            clip=5.0, seed=42, mix_prob=0.5, plateau_patience=5, lr_factor=0.5, early_stop_patience=15,
            dead_every=2, valve_rms=0.3, valve_rms_consec=2, valve_dead=0.9, valve_usdr=1.0, valve_usdr_consec=5,
            improve_smooth=1, valve_rms_at_epoch3=None)
F = dict(BASE, mask="sigmoid", plateau_patience=8, early_stop_patience=20, improve_smooth=3, valve_rms_at_epoch3=0.3)
CONFIGS = {"E": dict(BASE), "F23": dict(F, max_epochs=200), "F12": dict(F, L=512, S=128, max_epochs=200),
           "F6": dict(F, L=256, S=64, max_epochs=200)}


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


@torch.no_grad()
def validate(model, tracks, dev, dead, mask_act="relu", extra=False):
    model.eval(); set_tf32(False)
    cap = {}
    h = model.mask_conv.register_forward_hook(lambda m, i, o: cap.__setitem__("m", o)) if (dead or extra) else None
    rows = []
    for name in tracks:
        x, _ = load_track(name, "valid", dtype="float32")
        est = model(torch.from_numpy(x["mixture"].T.copy())[None].to(dev))[0].T.double().cpu().numpy()
        voc = x["vocals"].astype(np.float64)
        r = {"track": name, "usdr": usdr(voc, est), "sisdr": si_sdr(voc, est),
             "rms_ratio": float(np.sqrt(np.mean(est ** 2)) / np.sqrt(np.mean(voc ** 2)))}
        if dead or extra:
            lg = cap["m"][0]
            mk = torch.sigmoid(lg) if mask_act == "sigmoid" else torch.relu(lg)   # (N, frames)
            r["dead_frac"] = float((mk.amax(dim=1) <= 0).float().mean())
            r["mask_mean"] = float(mk.mean())
            if extra:
                r["mask_lt_0.01_frac"] = float((mk < 0.01).float().mean())
                nw = len(est) // 44100
                w = est[: nw * 44100].reshape(nw, 44100, -1)
                r["zero_output_1s_frac"] = float(np.mean(np.all(w == 0, axis=(1, 2))))
        rows.append(r)
    if h:
        h.remove()
    model.train(); set_tf32(True)
    f = lambda k: [r[k] for r in rows]
    out = {"usdr_median": float(np.median(f("usdr"))), "usdr_mean": float(np.mean(f("usdr"))),
           "sisdr_median": float(np.nanmedian(f("sisdr"))), "rms_ratio_median": float(np.median(f("rms_ratio"))),
           "rms_ratio_min": float(np.min(f("rms_ratio"))), "per_track": rows}
    if dead or extra:
        out.update(dead_frac_mean=float(np.mean(f("dead_frac"))), dead_frac_max=float(np.max(f("dead_frac"))),
                   mask_mean=float(np.mean(f("mask_mean"))))
    if extra:
        out.update(mask_lt_001_frac_mean=float(np.mean(f("mask_lt_0.01_frac"))),
                   zero_output_1s_frac_mean=float(np.mean(f("zero_output_1s_frac"))),
                   zero_output_1s_frac_max=float(np.max(f("zero_output_1s_frac"))))
    return out


def save_full(path, model, opt, st):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "model_cfg": model.cfg, "optimizer": opt.state_dict(), "state": st,
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "numpy": np.random.get_state(), "python": random.getstate()}}, tmp)
    os.replace(tmp, path)


def can_clear_stop(stop, epoch, cap):
    """Only an epoch-cap or time-budget stop may be continued; convergence / safety stops are final."""
    return epoch < cap and (stop == "max epochs reached" or str(stop or "").startswith("time budget reached"))


def check_config(previous, requested):
    for key in set(previous) | set(requested):
        if key != "max_epochs" and previous.get(key) != requested.get(key):
            raise ValueError(f"Resume changes the training config: {key}: {previous.get(key)!r} -> {requested.get(key)!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="F12", choices=list(CONFIGS))
    ap.add_argument("--out", required=True, help="output directory (checkpoints, train_log.json)")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time_budget_min", type=float, default=None,
                    help="stop cleanly (exit 4, resumable) before an epoch that would start after this many minutes")
    ap.add_argument("--max_epochs", type=int, default=None, help="override the epoch cap (e.g. when resuming)")
    ap.add_argument("--dryrun", action="store_true", help="tiny run (3 epochs, 1 chunk per track, 1 valid track)")
    add_data_args(ap)
    a = ap.parse_args(); apply_data_args(a)
    c = dict(CONFIGS[a.config])
    if a.max_epochs:
        c["max_epochs"] = a.max_epochs
    if a.dryrun:
        c.update(chunks_per_track=1, batch=4, max_epochs=3)
    os.makedirs(f"{a.out}/epochs", exist_ok=True)
    dev = torch.device(a.device); set_tf32(True); t0 = time.time()
    train, valid = read_split("train"), read_split("valid")
    if a.dryrun:
        valid = valid[:1]
    per_epoch = len(train) * c["chunks_per_track"]; spe = per_epoch // c["batch"]
    ds = MusdbChunks(train, "train", 6.0, items=per_epoch * c["max_epochs"], mix_prob=c["mix_prob"], seed=c["seed"])
    random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
    model = CausalConvTasNet(mask_act=c["mask"], **({"L": c["L"], "S": c["S"]} if "L" in c else {})).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    st = {"epoch": 0, "lr_base": c["lr"], "best_usdr": -1e9, "best_epoch": None, "since_best": 0, "since_lr_cut": 0,
          "rms_low_consec": 0, "usdr_low_consec": 0, "history": [], "lr_cuts": [], "stopped": None, "resumed_from": []}
    last = f"{a.out}/last.pt"
    if os.path.exists(last):
        with open(f"{a.out}/train_log.json") as previous_log:
            check_config(json.load(previous_log)["config"], c)
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck["model_cfg"] != model.cfg:
            raise ValueError("Resume model architecture differs from checkpoint")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); st = ck["state"]
        torch.set_rng_state(ck["rng"]["torch"].cpu())        # RNG states must be CPU ByteTensors
        if torch.cuda.is_available() and ck["rng"]["cuda"]:
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["rng"]["cuda"]])
        np.random.set_state(ck["rng"]["numpy"]); random.setstate(ck["rng"]["python"])
        st["resumed_from"].append({"epoch": st["epoch"], "previous_stop": st.get("stopped"), "max_epochs_now": c["max_epochs"]})
        if can_clear_stop(st.get("stopped"), st["epoch"], c["max_epochs"]):
            st["stopped"] = None
        print(f"resumed from epoch {st['epoch']}", flush=True)
    log = {"label": "valid results only (test is never used during training)", "config": c, "args": vars(a)}
    while st["epoch"] < c["max_epochs"] and not st["stopped"]:
        ep = st["epoch"] + 1
        if a.time_budget_min is not None and (time.time() - t0) / 60 > a.time_budget_min:
            st["stopped"] = f"time budget reached before epoch {ep} (resumable from last.pt)"
            save_full(last, model, opt, st)
            break
        sub = torch.utils.data.Subset(ds, range((ep - 1) * per_epoch, ep * per_epoch))
        dl = torch.utils.data.DataLoader(sub, batch_size=c["batch"], shuffle=False, num_workers=a.workers,
                                         drop_last=True, pin_memory=True, prefetch_factor=4 if a.workers else None)
        acc = {"loss": [], "gn": []}; te = time.time()
        for i, (mix, voc) in enumerate(dl):
            gstep = (ep - 1) * spe + i
            lr = st["lr_base"] * min(1.0, (gstep + 1) / (c["warmup_epochs"] * spe)) if ep <= c["warmup_epochs"] else st["lr_base"]
            for g in opt.param_groups:
                g["lr"] = lr
            mix, voc = mix.to(dev, non_blocking=True), voc.to(dev, non_blocking=True)
            loss = (model(mix) - voc).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip"]))
            if not (math.isfinite(float(loss)) and math.isfinite(gn)):
                st["stopped"] = f"non-finite loss/grad at epoch {ep} step {i}"; break
            opt.step(); acc["loss"].append(float(loss)); acc["gn"].append(gn)
        if st["stopped"]:
            break
        train_s = time.time() - te; tv = time.time()
        v = validate(model, valid, dev, dead=(ep % c["dead_every"] == 0), mask_act=c["mask"],
                     extra=(c["improve_smooth"] > 1 and ep % 2 == 0))
        rec = {"epoch": ep, "train_loss": float(np.mean(acc["loss"])), "grad_norm_median": float(np.median(acc["gn"])),
               "grad_norm_p90": float(np.percentile(acc["gn"], 90)), "lr_end": lr, "train_s": train_s,
               "valid_s": time.time() - tv, **{k: v[k] for k in v if k != "per_track"}, "per_track": v["per_track"],
               "elapsed_min_this_run": (time.time() - t0) / 60}
        torch.save({"model": model.state_dict(), "model_cfg": model.cfg, "epoch": ep}, f"{a.out}/epochs/ep{ep:03d}.pt")
        # best.pt = max RAW valid uSDR median; plateau / early stop use the moving average over improve_smooth epochs
        st.setdefault("best_ma", -1e9); st.setdefault("best_ma_epoch", None)
        if v["usdr_median"] > st["best_usdr"]:
            st.update(best_usdr=v["usdr_median"], best_epoch=ep)
            torch.save({"model": model.state_dict(), "model_cfg": model.cfg, "epoch": ep, "valid": rec}, f"{a.out}/best.pt")
        k = c["improve_smooth"]
        ma = float(np.mean([h["usdr_median"] for h in st["history"]][-(k - 1):] + [v["usdr_median"]])) if k > 1 else v["usdr_median"]
        rec["usdr_median_ma"] = ma
        if ma > st["best_ma"]:
            st.update(best_ma=ma, best_ma_epoch=ep, since_best=0, since_lr_cut=0)
        else:
            st["since_best"] += 1; st["since_lr_cut"] += 1
            if st["since_lr_cut"] >= c["plateau_patience"]:
                st["lr_base"] *= c["lr_factor"]; st["since_lr_cut"] = 0
                st["lr_cuts"].append({"after_epoch": ep, "new_lr": st["lr_base"]})
            if st["since_best"] >= c["early_stop_patience"]:
                st["stopped"] = f"early stop: {c['early_stop_patience']} epochs without improvement (after epoch {ep})"
        st["rms_low_consec"] = st["rms_low_consec"] + 1 if v["rms_ratio_median"] < c["valve_rms"] else 0
        st["usdr_low_consec"] = st["usdr_low_consec"] + 1 if v["usdr_median"] < c["valve_usdr"] else 0
        if st["rms_low_consec"] >= c["valve_rms_consec"]:
            st["stopped"] = f"SAFETY VALVE: RMS ratio < {c['valve_rms']} on {c['valve_rms_consec']} consecutive valids (epoch {ep})"
        if "dead_frac_mean" in v and v["dead_frac_mean"] > c["valve_dead"]:
            st["stopped"] = f"SAFETY VALVE: dead mask channels {v['dead_frac_mean']:.3f} > {c['valve_dead']} (epoch {ep})"
        if c["valve_rms_at_epoch3"] is not None and ep == 3 and v["rms_ratio_median"] < c["valve_rms_at_epoch3"]:
            st["stopped"] = f"SAFETY VALVE: RMS ratio {v['rms_ratio_median']:.3f} < {c['valve_rms_at_epoch3']} at epoch 3"
        if st["usdr_low_consec"] >= c["valve_usdr_consec"]:
            st["stopped"] = f"SAFETY VALVE: valid uSDR < {c['valve_usdr']} dB for {c['valve_usdr_consec']} consecutive epochs (epoch {ep})"
        st["history"].append(rec); st["epoch"] = ep
        save_full(last, model, opt, st)
        if ep % 10 == 0:
            save_full(f"{a.out}/snap_ep{ep:03d}.pt", model, opt, st)
        print(json.dumps({k: rec[k] for k in rec if k != "per_track"})
              + f" | best {st['best_epoch']} {st['best_usdr']:.3f} | best_ma {st['best_ma_epoch']} {st['best_ma']:.3f}"
              + (f" | STOP: {st['stopped']}" if st["stopped"] else ""), flush=True)
        log["state"] = {k: st[k] for k in st if k != "history"}; log["history"] = st["history"]
        dump_json(log, f"{a.out}/train_log.json")
    if not st["stopped"] and st["epoch"] >= c["max_epochs"]:
        st["stopped"] = "max epochs reached"
        save_full(last, model, opt, st)
    log["state"] = {k: st[k] for k in st if k != "history"}; log["history"] = st["history"]
    log["provenance"] = provenance(); dump_json(log, f"{a.out}/train_log.json")
    print("FINAL", json.dumps(log["state"], default=str), flush=True)
    stop = st["stopped"] or ""
    if stop.startswith("SAFETY VALVE") or stop.startswith("non-finite"):
        sys.exit(3)
    if stop.startswith("time budget"):
        sys.exit(4)


if __name__ == "__main__":
    main()
