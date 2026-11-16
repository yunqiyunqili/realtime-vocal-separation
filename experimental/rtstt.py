"""RT-STT (Real-Time Single-path TFC-TDF UNet; Wu et al., arXiv 2511.13146) — re-implementation for this repository.

Building blocks adapted from DTTNet-Pytorch (https://github.com/junyuchen-cjy/DTTNet-Pytorch,
commit 890a25e, Apache License 2.0; files src/dp_tdf/modules.py and src/dp_tdf/bandsequence.py, which in turn
credit BandSplitRNN-Pytorch). Modifications: everything is causal in time and stateful so that the
whole-signal path and the frame-by-frame streaming path run the SAME code.
The original Apache-2.0 license is in third_party/DTTNet/LICENSE.

Tensor layout inside the network: (B, C, T, F) — time before frequency, like DTTNet after its transpose.
Every time-dependent layer has signature  forward(x, state) -> (y, new_state); state=None means "start of
signal" (zeros), identical to left zero-padding.
"""
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCES = ["vocals", "drums", "bass", "other"]


class CausalConv2d(nn.Module):
    """Conv over (T, F): time kernel kt with ONLY past padding (kt-1 frames, cached when streaming);
    frequency kernel kf with symmetric 'same' padding (frequency is not causal-sensitive)."""

    def __init__(self, ci, co, kt=3, kf=3, bias=False, stride_f=1):
        super().__init__()
        self.kt, self.kf, self.stride_f = kt, kf, stride_f
        self.conv = nn.Conv2d(ci, co, (kt, kf), stride=(1, stride_f), padding=(0, kf // 2 if stride_f == 1 else 0), bias=bias)

    def forward(self, x, state: Optional[torch.Tensor] = None):
        if self.kt > 1:
            if state is None:
                state = x.new_zeros(x.shape[0], x.shape[1], self.kt - 1, x.shape[3])
            z = torch.cat([state, x], 2)
            new = z[:, :, -(self.kt - 1):]
        else:
            z, new = x, None
        return self.conv(z), new


class TFC(nn.Module):
    """DTTNet TFC: l x [conv k x k -> norm -> ReLU], here time-causal. Norm: BatchNorm2d (DTTNet default; in eval it
    is a fixed per-channel affine map, i.e. causal)."""

    def __init__(self, ci, co, l=2, k=3):
        super().__init__()
        self.convs = nn.ModuleList([CausalConv2d(ci if i == 0 else co, co, k, k) for i in range(l)])
        self.norms = nn.ModuleList([nn.BatchNorm2d(co) for _ in range(l)])

    def forward(self, x, state: Optional[List[Optional[torch.Tensor]]] = None):
        state = state or [None] * len(self.convs)
        new = []
        for conv, nm, s in zip(self.convs, self.norms, state):
            x, s2 = conv(x, s); x = F.relu(nm(x)); new.append(s2)
        return x, new


class TDF(nn.Module):
    """DTTNet TDF: Linear over frequency with bottleneck bn (f -> f/bn -> f), norm + ReLU after each. Per frame."""

    def __init__(self, c, f, bn=8):
        super().__init__()
        self.l1, self.n1 = nn.Linear(f, f // bn, bias=False), nn.BatchNorm2d(c)
        self.l2, self.n2 = nn.Linear(f // bn, f, bias=False), nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(self.n2(self.l2(F.relu(self.n1(self.l1(x))))))


class MediumTFCTDF(nn.Module):
    """DTTNet TFC_TDF_Res2 ('medium TFC-TDF' in RT-STT): res = TFC(1 conv); x = TFC1; x += TDF(x); x = TFC2; x += res."""

    def __init__(self, c, f, l=2, k=3, bn=8):
        super().__init__()
        self.tfc1, self.tfc2, self.res = TFC(c, c, l, k), TFC(c, c, l, k), TFC(c, c, 1, k)
        self.tdf = TDF(c, f, bn)

    def forward(self, x, state=None):
        s1, s2, sr = state if state is not None else (None, None, None)
        r, sr = self.res(x, sr)
        x, s1 = self.tfc1(x, s1)
        x = x + self.tdf(x)
        x, s2 = self.tfc2(x, s2)
        return x + r, (s1, s2, sr)


class RNNBlock(nn.Module):
    """DTTNet RNNModule across time, made causal: GroupNorm is computed PER FRAME over channel groups (DTTNet
    normalises over the whole time axis), LSTM is unidirectional, FC restores the size, residual."""

    def __init__(self, c, hidden, groups):
        super().__init__()
        self.gn = nn.GroupNorm(groups, c)
        self.rnn = nn.LSTM(c, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, c)

    def forward(self, x, state=None):
        B, C, T, Fq = x.shape
        z = x.permute(0, 3, 2, 1).reshape(B * Fq * T, C)            # one sample per (b, f, t)
        z = self.gn(z.unsqueeze(-1)).reshape(B * Fq, T, C)            # per-frame normalisation ((N, C, 1): ONNX-exportable)
        z, (h, c) = self.rnn(z, state)                                # sequences over time, one per (b, f)
        z = self.fc(z).reshape(B, Fq, T, C).permute(0, 3, 2, 1)
        return x + z, (h, c)


class SinglePath(nn.Module):
    """RT-STT single-path module: two RNN blocks over time (frequency is not modelled), residual inside each."""

    def __init__(self, c, hidden, groups):
        super().__init__()
        self.b1, self.b2 = RNNBlock(c, hidden, groups), RNNBlock(c, hidden, groups)

    def forward(self, x, state=None):
        s1, s2 = state if state is not None else (None, None)
        x, s1 = self.b1(x, s1); x, s2 = self.b2(x, s2)
        return x, (s1, s2)


class RTSTT(nn.Module):
    def __init__(self, n_fft=1024, hop=512, n_bins=384, g=16, L_sp=3, l=2, k=3, bn=8, lstm_hidden=None,
                 audio_ch=2, n_src=4, freq_ds=2):
        super().__init__()
        self.cfg = dict(n_fft=n_fft, hop=hop, n_bins=n_bins, g=g, L_sp=L_sp, l=l, k=k, bn=bn,
                        lstm_hidden=lstm_hidden, audio_ch=audio_ch, n_src=n_src, freq_ds=freq_ds)
        self.L, self.S, self.left = n_fft, hop, n_fft - hop        # same conventions as CausalConvTasNet
        self.n_fft, self.hop, self.Fb, self.ach, self.ns = n_fft, hop, n_bins, audio_ch, n_src
        C = 2 * audio_ch                                            # re/im x channels
        self.C, self.g = C, g
        c2, f2 = 2 * g, n_bins // freq_ds
        self.register_buffer("win", torch.hann_window(n_fft, periodic=True).sqrt(), persistent=False)
        # Explicit DFT / inverse-DFT matrices for the kept bins (ONNX opset 17 cannot export rfft/irfft; using the
        # same matrices in the offline and streaming paths keeps them exactly equivalent). Equivalent to
        # rfft(.)[..., :n_bins] and to irfft of the bins zero-padded to n_fft/2+1 (bin 0 imag ignored, as irfft).
        n = torch.arange(n_fft, dtype=torch.float64); kk = torch.arange(n_bins, dtype=torch.float64)
        ang = 2 * math.pi * kk[:, None] * n[None, :] / n_fft               # (Fb, n_fft)
        self.register_buffer("dft_c", torch.cos(ang).T.float().contiguous(), persistent=False)      # (n_fft, Fb)
        self.register_buffer("dft_s", (-torch.sin(ang)).T.float().contiguous(), persistent=False)
        wk = torch.full((n_bins, 1), 2.0 / n_fft, dtype=torch.float64); wk[0] = 1.0 / n_fft
        self.register_buffer("idft_c", (wk * torch.cos(ang)).float(), persistent=False)              # (Fb, n_fft)
        isn = -(wk * torch.sin(ang)); isn[0] = 0
        self.register_buffer("idft_s", isn.float(), persistent=False)
        # encoder
        self.first = nn.Conv2d(C, g, 1); self.first_n = nn.BatchNorm2d(g)
        self.enc = MediumTFCTDF(g, n_bins, l, k, bn)
        self.ds = nn.Conv2d(g, c2, (1, freq_ds), stride=(1, freq_ds)); self.ds_n = nn.BatchNorm2d(c2)
        # latent
        self.lat = MediumTFCTDF(c2, f2, l, k, bn)
        h = lstm_hidden or c2
        self.sp = nn.ModuleList([SinglePath(c2, h, max(1, c2 // 16)) for _ in range(L_sp)])
        self.expand = nn.Conv2d(c2, n_src * c2, 1)                  # "1x1 conv that expands to S sources"
        # decoder with channel-expansion fusion: all sources' channels in ONE mixed conv stack
        cd = n_src * g
        self.us = nn.ConvTranspose2d(n_src * c2, cd, (1, freq_ds), stride=(1, freq_ds)); self.us_n = nn.BatchNorm2d(cd)
        self.dec = MediumTFCTDF(cd, n_bins, l, k, bn)
        self.final = nn.Conv2d(cd, n_src * C, 1)

    # ---- STFT helpers (center=False, sqrt-Hann analysis and synthesis: periodic Hann is COLA at hop N/2) ----
    def _stft_frames(self, frames):                                  # frames (B, ch, T, n_fft) -> (B, C, T, Fb)
        w = frames * self.win
        re, im = w @ self.dft_c, w @ self.dft_s                      # (B, ch, T, Fb)
        B, ch, T, Fb = re.shape
        return torch.stack([re, im], 2).reshape(B, ch * 2, T, Fb)

    def _istft_frames(self, Y):                                      # Y (B, ns, C, T, Fb) -> frames (B, ns, ch, T, n_fft)
        B, ns, C, T, Fb = Y.shape
        Y = Y.reshape(B, ns, self.ach, 2, T, Fb)
        return (Y[:, :, :, 0] @ self.idft_c + Y[:, :, :, 1] @ self.idft_s) * self.win

    # ---- network on spectrogram frames ----
    def net(self, X, state=None):
        """X (B, C, T, Fb) -> mask logits (B, ns*C, T, Fb), new state."""
        st = state if state is not None else {}
        x = F.relu(self.first_n(self.first(X)))
        e, s_enc = self.enc(x, st.get("enc"))
        x = F.relu(self.ds_n(self.ds(e)))
        x, s_lat = self.lat(x, st.get("lat"))
        s_sp = []
        for i, sp in enumerate(self.sp):
            x, s = sp(x, st.get("sp", [None] * len(self.sp))[i]); s_sp.append(s)
        x = self.expand(x)
        x = F.relu(self.us_n(self.us(x)))
        x = x * e.repeat(1, self.ns, 1, 1)                           # skip: element-wise product (DTTNet), per source
        x, s_dec = self.dec(x, st.get("dec"))
        return self.final(x), {"enc": s_enc, "lat": s_lat, "sp": s_sp, "dec": s_dec}

    def _apply_mask(self, logits, X):
        B, _, T, Fb = logits.shape
        m = torch.softmax(logits.reshape(B, self.ns, self.C, T, Fb), dim=1)   # softmax over sources
        return m * X.unsqueeze(1)                                    # real mask on every re/im component

    # ---- offline (whole signal), causal ----
    def forward(self, x):
        """x (B, ch, T) -> estimates (B, ns, ch, T). Left pad n_fft-hop zeros; right pad flushes the last frames."""
        B, ch, T = x.shape
        Tr = math.ceil(T / self.hop) * self.hop
        xp = F.pad(x, (self.left, Tr - T + self.left))
        frames = xp.unfold(-1, self.n_fft, self.hop)                 # (B, ch, nT, n_fft)
        X = self._stft_frames(frames)
        logits, _ = self.net(X)
        fr = self._istft_frames(self._apply_mask(logits, X))         # (B, ns, ch, nT, n_fft)
        nT = fr.shape[3]
        y = F.fold(fr.reshape(B * self.ns * ch, nT, self.n_fft).transpose(1, 2), output_size=(1, (nT - 1) * self.hop + self.n_fft),
                   kernel_size=(1, self.n_fft), stride=(1, self.hop)).reshape(B, self.ns, ch, -1)
        return y[..., self.left:self.left + T]

    # ---- streaming: one hop (= one STFT frame) per call ----
    def step(self, block, state):
        """block (B, ch, hop). Output of call c = offline samples [c*hop - left, c*hop + hop - left)."""
        B, ch, _ = block.shape
        if state is None:
            state = {"inbuf": block.new_zeros(B, ch, self.left), "net": None,
                     "carry": block.new_zeros(B, self.ns, ch, self.n_fft - self.hop)}
        frame = torch.cat([state["inbuf"], block], -1).unsqueeze(2)  # (B, ch, 1, n_fft)
        X = self._stft_frames(frame)
        logits, net_state = self.net(X, state["net"])
        fr = self._istft_frames(self._apply_mask(logits, X))[:, :, :, 0]   # (B, ns, ch, n_fft)
        tot = F.pad(state["carry"], (0, self.hop)) + fr
        return tot[..., : self.hop], {"inbuf": frame[:, :, 0, self.hop:], "net": net_state, "carry": tot[..., self.hop:]}


def stream_process(m: RTSTT, x, block=None):
    """x (B, ch, T) -> delay-aligned (B, ns, ch, T) using m.step with the given block (must equal hop)."""
    block = block or m.hop
    assert block == m.hop
    B, ch, T = x.shape
    n = math.ceil((T + m.left) / block)
    xp = F.pad(x, (0, n * block - T))
    st, outs = None, []
    for c in range(n):
        o, st = m.step(xp[..., c * block:(c + 1) * block], st)
        outs.append(o)
    return torch.cat(outs, -1)[..., m.left:m.left + T]
