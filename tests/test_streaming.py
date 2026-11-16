"""Fast CPU checks on randomly initialised models (no data, no checkpoints; < 1 minute).

1. Causality probe: perturbing the input from sample t0 on must change the offline output first at exactly
   t0 - ((L - S) + t0 mod S), never earlier (so the worst-case lookahead is L - 1 samples).
2. Streaming consistency: offline whole-signal output == block-streaming output (delay-aligned) within 1e-4,
   for the reference per-hop path, the deployment StreamCore (eager, block 512) and its ONNX Runtime export.
3. The same streaming check for the two experimental 11.6 ms comparison models (C and S)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtvs.checks import block_consistency, future_probe, random_input, step_consistency  # noqa: E402
from rtvs.inference import make_stream  # noqa: E402
from rtvs.model import CausalConvTasNet  # noqa: E402
from rtvs.stream import run_stream  # noqa: E402

TIERS = {"F12": dict(L=512, S=128), "F6": dict(L=256, S=64)}


def model(tier):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    return CausalConvTasNet(mask_act="sigmoid", **TIERS[tier]).eval()


@pytest.mark.parametrize("tier", list(TIERS))
def test_causal_probe_matches_theory(tier):
    m = model(tier)
    r = future_probe(m, random_input(1.0))
    assert r["causal_within_window"]
    assert r["all_match_theory"], [x for x in r["rows"] if x["lookahead"] != x["theory"]]
    assert r["lookahead_max"] <= m.L - 1


@pytest.mark.parametrize("tier", list(TIERS))
def test_reference_step_matches_offline(tier):
    assert step_consistency(model(tier), random_input(1.0))["pass"]


@pytest.mark.parametrize("impl", ["eager", "onnxruntime"])
def test_block512_streaming_matches_offline(impl):
    if impl == "onnxruntime":
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
    m = model("F12")
    step, init = make_stream(m, "convtasnet", 512, impl, 1)
    r = block_consistency(m, step, init, random_input(2.0)[0], 512)
    assert r["pass"], r


def test_output_delay_f12():
    m = model("F12")
    assert m.left == 384 and abs((m.L - 1) / 44100 * 1e3 - 11.587) < 1e-3


@pytest.mark.parametrize("arch", ["music_fft512", "music_asym"])
def test_experimental_streaming_matches_offline(arch):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    if arch == "music_fft512":
        from experimental.fast_music import FastMusic
        m = FastMusic("fft512").eval()
    else:
        from experimental.asym_music import AsymMusic
        m = AsymMusic().eval()
    x = random_input(0.5)[0] * 0.5
    with torch.no_grad():
        ref = m(x[None])[0]
        st = m.init_state(); ys = []
        xp = torch.nn.functional.pad(x[None], (0, m.L))
        for i in range(0, xp.shape[-1] - m.H + 1, m.H):
            y, st = m.step(xp[:, :, i:i + m.H], st); ys.append(y)
        est = torch.cat(ys, -1)[0, :, m.left:m.left + x.shape[-1]]
    assert float((ref - est).abs().max()) < 1e-4
    if arch == "music_fft512":
        pytest.importorskip("onnxruntime")
        step, init = make_stream(m, arch, 512, "onnxruntime", 1)
        with torch.no_grad():
            est2, _ = run_stream(step, init(), x, 512, left=m.left)
        assert float((ref - est2).abs().max()) < 1e-4
