"""Causal Conv-TasNet for stereo vocals extraction, with an exact block-streaming twin.

Offline (training / reference) path:  CausalConvTasNet.forward(x)  x: (B, 2, T) -> (B, 2, T)
Reference streaming path:              CausalConvTasNet.step(block, state)  block: (B, 2, S)
Deployment streaming path:             rtvs.stream.StreamCore (block = any multiple of S, ONNX-exportable)

Framing, shown for the defaults L=1024, S=256 (F23); F12 uses L=512, S=128 and F6 L=256, S=64,
with every number below scaled accordingly (output delay = L - S samples):
  The signal is left-padded with L-S = 768 zeros, so encoder frame t covers original samples
  [256 t - 768, 256 t + 256).  Frame t can be computed as soon as input block t
  (samples 256t .. 256t+255) has arrived.  The decoder (ConvTranspose1d, i.e. overlap-add of
  1024-sample frames, hop 256) produces output sample n from frames t = floor(n/256) .. +3, so
  sample n is final after block floor(n/256)+3 has arrived: lookahead = 1023 - (n mod 256),
  i.e. 768..1023 samples; the streaming output is the offline output delayed by 768 samples
  (3 blocks).
  Offline right padding: 768 zeros after rounding T up to a multiple of 256, so every frame
  that touches a real sample is computed (equivalent to flushing the stream with 3 zero blocks).

Normalisation: cumulative layer norm (cLN), statistics over channels and all frames <= t only.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class CumLN(nn.Module):
    """Cumulative layer norm (Luo & Mesgarani 2019, causal config). x: (B, C, T)."""

    def __init__(self, C, eps=EPS):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, C, 1))
        self.beta = nn.Parameter(torch.zeros(1, C, 1))
        self.eps = eps
        self.C = C

    def _norm(self, x, cum_sum, cum_pow, cnt):
        mean = cum_sum / cnt
        var = ((cum_pow - 2 * mean * cum_sum) / cnt + mean ** 2).clamp(min=0)  # guard float32 cancellation
        return (x - mean.unsqueeze(1)) / (var.unsqueeze(1) + self.eps).sqrt() * self.gamma + self.beta

    def forward(self, x):
        B, C, T = x.shape
        cum_sum = torch.cumsum(x.sum(1), dim=1)          # (B, T)
        cum_pow = torch.cumsum(x.pow(2).sum(1), dim=1)
        cnt = torch.arange(C, C * (T + 1), C, device=x.device, dtype=x.dtype).view(1, T)
        return self._norm(x, cum_sum, cum_pow, cnt)

    def step(self, x, state):
        """x: (B, C, 1). state: (sum, pow, count) or None."""
        s, p = x.sum(1), x.pow(2).sum(1)                   # (B, 1)
        if state is None:
            cs, cp, n = s, p, 1
        else:
            cs, cp, n = state[0] + s, state[1] + p, state[2] + 1
        cnt = torch.full_like(cs, float(self.C * n))
        return self._norm(x, cs, cp, cnt), (cs, cp, n)


class TCNBlock(nn.Module):
    def __init__(self, B, H, Sc, P, d):
        super().__init__()
        self.conv_in = nn.Conv1d(B, H, 1)
        self.act1 = nn.PReLU()
        self.norm1 = CumLN(H)
        self.dw = nn.Conv1d(H, H, P, dilation=d, groups=H)
        self.act2 = nn.PReLU()
        self.norm2 = CumLN(H)
        self.res = nn.Conv1d(H, B, 1)
        self.skip = nn.Conv1d(H, Sc, 1)
        self.pad = (P - 1) * d

    def forward(self, x):
        y = self.norm1(self.act1(self.conv_in(x)))
        y = F.pad(y, (self.pad, 0))                        # causal: past only
        y = self.norm2(self.act2(self.dw(y)))
        return x + self.res(y), self.skip(y)

    def step(self, x, st):
        st = st or {}
        y, st_n1 = self.norm1.step(self.act1(self.conv_in(x)), st.get("n1"))
        cache = st.get("cache")
        if cache is None:
            cache = x.new_zeros(x.shape[0], y.shape[1], self.pad)
        z = torch.cat([cache, y], dim=2)
        new_cache = z[:, :, 1:] if self.pad > 0 else cache
        y2, st_n2 = self.norm2.step(self.act2(self.dw(z)), st.get("n2"))
        return x + self.res(y2), self.skip(y2), {"n1": st_n1, "n2": st_n2, "cache": new_cache}


class CausalConvTasNet(nn.Module):
    def __init__(self, N=512, L=1024, S=256, B=128, H=512, Sc=128, P=3, X=8, R=3,
                 in_ch=2, out_ch=2, mask_act="sigmoid"):
        super().__init__()
        self.cfg = dict(N=N, L=L, S=S, B=B, H=H, Sc=Sc, P=P, X=X, R=R, in_ch=in_ch, out_ch=out_ch, mask_act=mask_act)
        self.L, self.S, self.N = L, S, N
        self.left = L - S
        self.encoder = nn.Conv1d(in_ch, N, L, stride=S, bias=False)
        self.ln = CumLN(N)
        self.bottleneck = nn.Conv1d(N, B, 1)
        self.blocks = nn.ModuleList([TCNBlock(B, H, Sc, P, 2 ** x) for _ in range(R) for x in range(X)])
        self.out_act = nn.PReLU()
        self.mask_conv = nn.Conv1d(Sc, N, 1)
        self.mask_act = mask_act
        self.decoder = nn.ConvTranspose1d(N, out_ch, L, stride=S, bias=False)

    # ---- shared pieces -------------------------------------------------------
    def _mask(self, m):
        return torch.sigmoid(m) if self.mask_act == "sigmoid" else F.relu(m)

    def receptive_field(self):
        P, X, R = self.cfg["P"], self.cfg["X"], self.cfg["R"]
        frames = 1 + R * sum((P - 1) * 2 ** x for x in range(X))
        past_samples = (frames - 1) * self.S + (self.L - self.S)  # beyond the current block
        return {"tcn_frames": frames, "conv_rf_samples_total": (frames - 1) * self.S + self.L,
                "past_context_samples": past_samples,
                "note": "convolutional RF only; cLN statistics accumulate over the entire past"}

    # ---- offline -------------------------------------------------------------
    def forward(self, x):
        Bsz, C, T = x.shape
        Tr = math.ceil(T / self.S) * self.S
        xp = F.pad(x, (self.left, Tr - T + self.left))
        w = F.relu(self.encoder(xp))                        # (B, N, F)
        y = self.bottleneck(self.ln(w))
        skip = 0
        for blk in self.blocks:
            y, s = blk(y)
            skip = skip + s
        m = self._mask(self.mask_conv(self.out_act(skip)))
        out = self.decoder(w * m)                           # (B, out_ch, (F-1)S + L)
        return out[:, :, self.left:self.left + T]

    # ---- streaming -----------------------------------------------------------
    def step(self, block, state):
        """block: (B, in_ch, S). Returns (out (B, out_ch, S), state).
        Output of call #b is offline output samples [S(b-3), S(b-3)+S) (zeros-padded region for b<3)."""
        Bsz = block.shape[0]
        if state is None:
            state = {"inbuf": block.new_zeros(Bsz, block.shape[1], self.left), "ln": None,
                     "blocks": [None] * len(self.blocks),
                     "ola": block.new_zeros(Bsz, self.cfg["out_ch"], self.L)}
        frame = torch.cat([state["inbuf"], block], dim=2)   # (B, C, L)
        w = F.relu(self.encoder(frame))                     # (B, N, 1)
        ln_out, ln_st = self.ln.step(w, state["ln"])
        y = self.bottleneck(ln_out)
        skip = 0
        new_bst = []
        for blk, st in zip(self.blocks, state["blocks"]):
            y, s, st = blk.step(y, st)
            skip = skip + s
            new_bst.append(st)
        m = self._mask(self.mask_conv(self.out_act(skip)))
        dec = self.decoder(w * m)                           # (B, out_ch, L)
        ola = state["ola"] + dec
        out = ola[:, :, :self.S]
        new_ola = torch.cat([ola[:, :, self.S:], ola.new_zeros(Bsz, ola.shape[1], self.S)], dim=2)
        return out, {"inbuf": frame[:, :, self.S:], "ln": ln_st, "blocks": new_bst, "ola": new_ola}


def stream_process(model, x, block=256):
    """Run the streaming path over a whole signal x (B, C, T); returns output aligned with the
    offline output (B, out_ch, T) (i.e. the 768-sample algorithmic delay removed) and the
    raw per-block outputs.  Feeds ceil(T/256) blocks + 3 zero blocks (flush)."""
    assert block == model.S, "streaming block must equal the encoder hop"
    Bsz, C, T = x.shape
    nb = math.ceil(T / block)
    xp = F.pad(x, (0, nb * block - T + model.left))
    st, outs = None, []
    for b in range(nb + model.left // block):
        o, st = model.step(xp[:, :, b * block:(b + 1) * block], st)
        outs.append(o)
    raw = torch.cat(outs, dim=2)
    return raw[:, :, model.left:model.left + T], raw


@torch.no_grad()
def count_macs(model, seconds=1.0, sr=44100):
    """Multiply-accumulates of conv layers for `seconds` of stereo audio via forward hooks
    (elementwise ops: PReLU, cLN, mask, residual adds are not counted)."""
    macs = {"total": 0}

    def hook(mod, inp, out):
        if isinstance(mod, nn.ConvTranspose1d):
            n = inp[0].shape[1] * mod.out_channels // mod.groups * mod.kernel_size[0] * inp[0].shape[2]
        else:
            n = out.shape[1] * (mod.in_channels // mod.groups) * mod.kernel_size[0] * out.shape[2]
        n *= out.shape[0]
        macs["total"] += n

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d))]
    T = int(seconds * sr)
    T = (T // model.S) * model.S
    # count on the frame-exact streaming path: one frame per 256 samples
    st = None
    x = torch.zeros(1, model.cfg["in_ch"], T)
    for b in range(T // model.S):
        _, st = model.step(x[:, :, b * model.S:(b + 1) * model.S], st)
    for h in hs:
        h.remove()
    return macs["total"] / (T / sr)  # MACs per second of audio
