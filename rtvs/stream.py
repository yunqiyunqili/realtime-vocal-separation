"""Deployment streaming core for CausalConvTasNet (batch 1, time-major 2-D tensors).

Shapes below are for L=1024, S=256 (F23); for other tiers replace 256 by S and 768 by L - S.
* Per-layer state caching: each call computes ONLY the k = block/S new encoder frames.
  State (all tensors, fixed shapes):
    inbuf   (2, 768)      last L - S input samples (past only)
    carry   (2, 768)      decoder overlap-add tail not yet final
    ln      (n_ln, 3)     cLN running (sum, sum of squares, count) for every cLN layer
    caches  24 x (2d, H)  depthwise-conv history (exactly the (P-1)*d past frames needed)
* 1x1 convs become matmuls on (k, C); res+skip convs fused into one matmul; PReLU via where.
* Exportable: plain eager, torch.jit.script, and ONNX (via FlatCore, block size fixed per export).
Output of call c = offline output samples [c*B - (L-S), c*B + B - (L-S))  (fixed L - S sample delay).
"""
import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CausalConvTasNet


class StreamCore(nn.Module):
    def __init__(self, m: CausalConvTasNet, block: int = 256):
        super().__init__()
        c = m.cfg
        assert c["P"] == 3 and block % m.S == 0
        self.S, self.L, self.block, self.k = m.S, m.L, block, block // m.S
        self.left = m.L - m.S
        self.N, self.H, self.B, self.Sc = c["N"], c["H"], c["B"], c["Sc"]
        self.out_ch = c["out_ch"]
        self.nblk = len(m.blocks)
        self.dil: List[int] = [blk.pad // 2 for blk in m.blocks]
        P = lambda t: nn.Parameter(t.detach().clone().contiguous(), requires_grad=False)
        self.We = P(m.encoder.weight.reshape(self.N, -1).t())          # (2L, N)
        self.ln0_g, self.ln0_b = P(m.ln.gamma.reshape(-1)), P(m.ln.beta.reshape(-1))
        self.Wbn, self.bbn = P(m.bottleneck.weight[:, :, 0].t()), P(m.bottleneck.bias)
        Win, bin_, a1, g1, b1, wdw, bdw, a2, g2, b2, Wrs, brs = ([] for _ in range(12))
        for blk in m.blocks:
            Win.append(blk.conv_in.weight[:, :, 0].t()); bin_.append(blk.conv_in.bias)
            a1.append(blk.act1.weight.reshape(1)); g1.append(blk.norm1.gamma.reshape(-1)); b1.append(blk.norm1.beta.reshape(-1))
            wdw.append(blk.dw.weight[:, 0, :].t()); bdw.append(blk.dw.bias)            # (3, H)
            a2.append(blk.act2.weight.reshape(1)); g2.append(blk.norm2.gamma.reshape(-1)); b2.append(blk.norm2.beta.reshape(-1))
            Wrs.append(torch.cat([blk.res.weight[:, :, 0], blk.skip.weight[:, :, 0]], 0).t())  # (H, B+Sc)
            brs.append(torch.cat([blk.res.bias, blk.skip.bias], 0))
        st = lambda l: P(torch.stack(l))
        self.Win, self.bin, self.a1, self.g1, self.b1 = st(Win), st(bin_), st(a1), st(g1), st(b1)
        self.wdw, self.bdw, self.a2, self.g2, self.b2 = st(wdw), st(bdw), st(a2), st(g2), st(b2)
        self.Wrs, self.brs = st(Wrs), st(brs)
        self.aout = P(m.out_act.weight.reshape(1))
        self.Wm, self.bm = P(m.mask_conv.weight[:, :, 0].t()), P(m.mask_conv.bias)
        self.sigmoid_mask = c["mask_act"] == "sigmoid"
        self.Wd = P(m.decoder.weight.reshape(self.N, -1))                 # (N, out_ch*L)
        self.eps = 1e-8
        self.register_buffer("cnt_steps", torch.arange(1, self.k + 1, dtype=torch.float32), persistent=False)

    # -- state ----------------------------------------------------------------
    def init_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        inbuf = torch.zeros(2, self.left)
        carry = torch.zeros(self.out_ch, self.left)
        ln = torch.zeros(1 + 2 * self.nblk, 3)
        caches = [torch.zeros(2 * d, self.H) for d in self.dil]
        return inbuf, carry, ln, caches

    def _cln(self, x: torch.Tensor, st: torch.Tensor, g: torch.Tensor, b: torch.Tensor):
        C = x.shape[1]
        cs = st[0] + torch.cumsum(x.sum(1), 0)
        cp = st[1] + torch.cumsum((x * x).sum(1), 0)
        cnt = st[2] + C * self.cnt_steps
        mean = cs / cnt
        var = ((cp - 2 * mean * cs) / cnt + mean * mean).clamp(min=0)  # guard float32 cancellation
        y = (x - mean.unsqueeze(1)) * torch.rsqrt(var + self.eps).unsqueeze(1) * g + b
        return y, torch.stack([cs[-1], cp[-1], cnt[-1]])

    @staticmethod
    def _prelu(x: torch.Tensor, a: torch.Tensor):
        return torch.where(x >= 0, x, a * x)

    def forward(self, block: torch.Tensor, inbuf: torch.Tensor, carry: torch.Tensor, ln: torch.Tensor,
                caches: List[torch.Tensor]):
        """block (2, B) -> out (2, B) plus new state."""
        buf = torch.cat([inbuf, block], 1)                                # (2, 768 + B)
        frames = buf.unfold(1, self.L, self.S).permute(1, 0, 2).reshape(self.k, -1)  # (k, 2L)
        w = torch.relu(frames @ self.We)                                  # (k, N)
        new_ln: List[torch.Tensor] = []
        y, s = self._cln(w, ln[0], self.ln0_g, self.ln0_b); new_ln.append(s)
        y = y @ self.Wbn + self.bbn                                       # (k, B)
        skip = torch.zeros(self.k, self.Sc, device=y.device, dtype=y.dtype)  # follow the input device (CPU or GPU)
        new_caches: List[torch.Tensor] = []
        for i, d in enumerate(self.dil):
            h = self._prelu(y @ self.Win[i] + self.bin[i], self.a1[i])
            h, s = self._cln(h, ln[1 + 2 * i], self.g1[i], self.b1[i]); new_ln.append(s)
            z = torch.cat([caches[i], h], 0)                              # (2d + k, H)
            wd = self.wdw[i]
            h = wd[0] * z[0:self.k] + wd[1] * z[d:d + self.k] + wd[2] * z[2 * d:2 * d + self.k] + self.bdw[i]
            new_caches.append(z[self.k:])
            h = self._prelu(h, self.a2[i])
            h, s = self._cln(h, ln[2 + 2 * i], self.g2[i], self.b2[i]); new_ln.append(s)
            rs = h @ self.Wrs[i] + self.brs[i]
            y = y + rs[:, :self.B]
            skip = skip + rs[:, self.B:]
        mk = self._prelu(skip, self.aout) @ self.Wm + self.bm
        mk = torch.sigmoid(mk) if self.sigmoid_mask else torch.relu(mk)
        dec = ((w * mk) @ self.Wd).reshape(self.k, self.out_ch, self.L)   # (k, 2, L)
        tot = F.pad(carry, (0, self.block))                               # (2, B + 768)
        for j in range(self.k):
            tot = tot + F.pad(dec[j], (j * self.S, (self.k - 1 - j) * self.S))
        out = tot[:, :self.block]
        return out, buf[:, self.block:], tot[:, self.block:], torch.stack(new_ln), new_caches


class FlatCore(nn.Module):
    """Flat-argument wrapper for ONNX export."""

    def __init__(self, core: StreamCore):
        super().__init__()
        self.core = core

    def forward(self, block, inbuf, carry, ln, *caches):
        out, ib, ca, l2, cs = self.core(block, inbuf, carry, ln, list(caches))
        return (out, ib, ca, l2, *cs)


def run_stream(step_fn, init_state, x: torch.Tensor, block: int, left: int = 768):
    """x (2, T). step_fn(block, state) -> (out, state). Returns delay-aligned (2, T) output
    and per-call wall times (s)."""
    import time
    T = x.shape[1]
    n = math.ceil((T + left) / block)
    xp = F.pad(x, (0, n * block - T))
    st, outs, times = init_state, [], []
    for c in range(n):
        blk = xp[:, c * block:(c + 1) * block]
        t = time.perf_counter()
        o, st = step_fn(blk, st)
        times.append(time.perf_counter() - t)
        outs.append(o)
    raw = torch.cat(outs, 1)
    return raw[:, left:left + T], times
