"""Checkpoint loading and block-streaming inference (eager PyTorch or ONNX Runtime).

All reported numbers use ONNX Runtime on CPU, 1 intra-op / 1 inter-op thread, sequential execution, block 512.
A streaming "step" is a function (block (2, 512) tensor, state) -> (out (2, 512) tensor, state); together with
``rtvs.stream.run_stream`` it produces the delay-aligned whole-signal output."""
import io
import os
import tempfile

import torch

from .model import CausalConvTasNet
from .stream import FlatCore, StreamCore

ARCHS = ("convtasnet", "music_fft512", "music_asym")


def ort_session(model_bytes_or_path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_bytes_or_path, so, providers=["CPUExecutionProvider"])


def detect_arch(ck):
    cfg = ck.get("model_cfg") or {}
    if "N" in cfg:
        return "convtasnet"
    variant = cfg.get("variant")
    if variant == "fft512":
        return "music_fft512"
    if variant == "asym":
        return "music_asym"
    raise ValueError(f"cannot infer the architecture from model_cfg={cfg!r}; pass --arch")


def load_checkpoint(path, arch=None):
    """Returns (model in eval mode, arch). Experimental architectures live in experimental/."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = arch or detect_arch(ck)
    if arch == "convtasnet":
        m = CausalConvTasNet(**ck["model_cfg"])
    elif arch == "music_fft512":
        from experimental.fast_music import FastMusic
        m = FastMusic("fft512")
    elif arch == "music_asym":
        from experimental.asym_music import AsymMusic
        m = AsymMusic()
    else:
        raise ValueError(arch)
    m.load_state_dict(ck["model"])
    return m.eval(), arch


def export_convtasnet_onnx(m, block=512, path=None):
    """Export StreamCore(m, block) with explicit state inputs/outputs (opset 17). Returns (bytes or path, names)."""
    core = StreamCore(m, block).eval()
    ib, ca, ln, cs = core.init_state()
    names = ["block", "inbuf", "carry", "ln"] + [f"c{i}" for i in range(len(cs))]
    onames = ["out", "inbuf_out", "carry_out", "ln_out"] + [f"c{i}_out" for i in range(len(cs))]
    target = path or io.BytesIO()
    torch.onnx.export(FlatCore(core), (torch.zeros(2, block), ib, ca, ln, *cs), target, input_names=names,
                      output_names=onames, opset_version=17, dynamo=False)
    init = [ib.numpy(), ca.numpy(), ln.numpy()] + [c.numpy() for c in cs]
    return (path if path else target.getvalue()), names, init, core


def make_stream(m, arch="convtasnet", block=512, impl="onnxruntime", threads=1):
    """Returns (step, init_fn). impl: 'onnxruntime' (used for every reported number) or 'eager'."""
    torch.set_num_threads(threads)
    if arch == "convtasnet":
        if impl == "eager":
            core = StreamCore(m, block).eval()

            def step(b, st):
                o, *st = core(b, *st)
                return o, st
            return step, lambda: list(core.init_state())
        model, names, init, _ = export_convtasnet_onnx(m, block)
        sess = ort_session(model, threads)

        def step(b, st):
            r = sess.run(None, dict(zip(names, [b.numpy()] + st)))
            return torch.from_numpy(r[0]), r[1:]
        return step, lambda: [x.copy() for x in init]
    if block != 512 or impl != "onnxruntime":
        raise ValueError("experimental models are exported for ONNX Runtime with block 512 only")
    onnx_path = os.path.join(tempfile.mkdtemp(prefix="rtvs_"), "model.onnx")
    if arch == "music_fft512":
        from experimental.fast_music_ort import make
    else:
        from experimental.asym_music import make
    return make(m, onnx_path)
