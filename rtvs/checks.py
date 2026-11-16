"""Causality probe, streaming-consistency checks, parameter and MAC counts (CPU, float32).

future_probe: perturb the input only from sample t0 on and find the earliest output sample that changes.
  measured lookahead = t0 - first changed sample. For a randomly initialised Conv-TasNet every probe point must
  match the theory exactly: (L - S) + (t0 mod S), i.e. between L - S and L - 1 samples; the worst case L - 1 is the
  algorithmic latency ((L - 1) / 44100 s; 11.59 ms for F12). For a trained model points whose output is exactly
  zero around t0 cannot be measured and are reported separately.
consistency: offline whole-signal output vs block-streaming output (delay-aligned), tolerance 1e-4."""
import torch

from .model import CausalConvTasNet, count_macs, stream_process
from .stream import run_stream

TOL = 1e-4


@torch.no_grad()
def future_probe(m: CausalConvTasNet, x, seed=1, positions=None):
    L, S = m.L, m.S
    y0 = m(x)
    g = torch.Generator().manual_seed(seed)
    T = x.shape[-1]
    positions = positions or [T // 4, T // 2, 3 * T // 4]
    phases = sorted({0, 1, S // 4, S // 2 - 1, S // 2, 3 * S // 4, S - 2, S - 1})
    rows = []
    for pos in positions:
        k = pos // S
        for p in phases:
            t0 = S * k + p
            xp = x.clone()
            xp[:, :, t0:] += 0.1 * torch.randn(xp[:, :, t0:].shape, generator=g)
            d = (m(xp) - y0).abs().amax(dim=(0, 1))
            nz = torch.nonzero(d > 0)
            first = int(nz[0]) if len(nz) else None
            zero_win = bool((y0[:, :, max(0, t0 - (L - 1)):t0 + 1].abs().max() == 0).item())
            rows.append({"t0": t0, "phase": p, "first_changed": first,
                         "lookahead": (t0 - first) if first is not None else None, "theory": (L - S) + p,
                         "no_change_before_window": first is None or first >= t0 - (L - 1),
                         "output_zero": zero_win or first is None})
    la = [r["lookahead"] for r in rows if r["lookahead"] is not None]
    return {"rows": rows, "lookahead_min": min(la) if la else None, "lookahead_max": max(la) if la else None,
            "all_match_theory": all(r["lookahead"] == r["theory"] for r in rows),
            "causal_within_window": all(r["no_change_before_window"] for r in rows),
            "algorithmic_latency_samples": L - 1, "algorithmic_latency_ms": (L - 1) / 44100 * 1e3}


@torch.no_grad()
def step_consistency(m: CausalConvTasNet, x):
    """Offline forward vs the reference per-hop streaming path (model.step)."""
    y = m(x)
    ys, _ = stream_process(m, x, m.S)
    err = float((y - ys).abs().max())
    return {"max_abs_err": err, "pass": err <= TOL}


@torch.no_grad()
def block_consistency(m, step, init, x, block):
    """Offline forward vs any block-streaming step function (StreamCore eager / ONNX Runtime). x: (2, T)."""
    y = m(x[None])[0]
    ys, _ = run_stream(step, init(), x, block, left=m.left)
    err = float((y - ys).abs().max())
    return {"block": block, "max_abs_err": err, "pass": err <= TOL}


def model_stats(m: CausalConvTasNet):
    return {"params": int(sum(p.numel() for p in m.parameters())),
            "gmac_per_audio_second": count_macs(m) / 1e9,
            "algorithmic_latency_ms": (m.L - 1) / 44100 * 1e3,
            "output_delay_samples": m.left, "receptive_field": m.receptive_field()}


def random_input(seconds=2.0, seed=0, sr=44100):
    g = torch.Generator().manual_seed(seed)
    return 0.1 * torch.randn(1, 2, int(seconds * sr), generator=g)

