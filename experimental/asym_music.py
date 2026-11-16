"""MusicSparse v2 with an asymmetric analysis/synthesis STFT window pair (low-delay design in the spirit of
Mauler & Martin 2007). Only the STFT windows (and the hop) differ from MusicSparse v2; the network (regions,
encoders, cross-band conv, local/global GRUs, decoders, mask) is MusicSparse v2 unchanged.

  N  = 1024  analysis window / FFT length (frequency resolution of v2: 513 bins, v2 band regions)
  H  = 128   hop (same as variant C)
  Ms = 512   synthesis window length; non-zero only on the LAST 512 samples of each frame
  analysis  w_a: rising sqrt-Hann over the first N - Ms/2 = 768 samples (long, flat rise), then the falling half of
            a sqrt-Hann of length Ms over the last Ms/2 = 256 samples (short right side, matched to synthesis)
  synthesis w_s: zero on [0, N - Ms); on [N - Ms, N): w_s = p / w_a with p = Hann_Ms(periodic) / 2
  => w_a * w_s = p on the last 512 samples, and sum_t p(n - tH) = 1 for H = 128 (periodic Hann, 75% overlap)
     -> perfect reconstruction without envelope division (the OLA envelope division is kept, it equals 1).
Algorithmic latency is set by the synthesis support (last 512 samples), i.e. the same as variant C (FFT 512).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from experimental.music_sparse_v2 import MusicSparseV2


def asym_windows(N=1024, Ms=512):
    rise = N - Ms // 2                                              # 768
    long_hann = torch.hann_window(2 * rise, periodic=True, dtype=torch.float64)
    short_hann = torch.hann_window(Ms, periodic=True, dtype=torch.float64)
    wa = torch.empty(N, dtype=torch.float64)
    wa[:rise] = long_hann[:rise].sqrt()                            # 0 -> 1 over 768 samples
    wa[rise:] = short_hann[Ms // 2:].sqrt()                        # 1 -> 0 over 256 samples
    p = torch.zeros(N, dtype=torch.float64); p[N - Ms:] = short_hann / 2
    ws = torch.zeros(N, dtype=torch.float64)
    nz = slice(N - Ms, N)
    ws[nz] = torch.where(wa[nz] > 0, p[nz] / wa[nz].clamp_min(1e-12), torch.zeros_like(p[nz]))
    return wa.float(), ws.float(), p.float()


class AsymMusic(MusicSparseV2):
    L = 1024; H = 128; left = 1024 - 128                           # frame t covers original [tH - 896, tH + 128)

    def __init__(self):
        super().__init__()
        self.variant = 'asym'
        wa, ws, p = asym_windows(self.L, 512)
        self.register_buffer('win_a', wa); self.register_buffer('win_s', ws); self.register_buffer('win_p', wa * ws)
        # keep the inherited buffer name 'win' (= analysis window) so generic helpers do not break
        self.win = wa.clone()

    def forward(self, x):
        B, C, N = x.shape
        xp = F.pad(x, (self.left, self.L)); fr = xp.unfold(-1, self.L, self.H)
        z = torch.fft.rfft(fr * self.win_a); y, _ = self.spectral(z)
        f = torch.fft.irfft(y, n=self.L) * self.win_s; T = f.shape[-2]; length = (T - 1) * self.H + self.L
        raw = F.fold(f.reshape(B * C, T, self.L).transpose(1, 2), (1, length), (1, self.L), stride=(1, self.H)).reshape(B, C, length)
        den = F.fold(self.win_p[None, :, None].expand(1, -1, T), (1, length), (1, self.L), stride=(1, self.H)).reshape(1, 1, length)
        return (raw / den.clamp_min(1e-8))[:, :, self.left:self.left + N]

    def stft_istft_identity(self, x):
        """No network: STFT (analysis window) -> iSTFT (synthesis window) -> OLA / envelope, same alignment."""
        B, C, N = x.shape
        xp = F.pad(x, (self.left, self.L)); fr = xp.unfold(-1, self.L, self.H)
        f = torch.fft.irfft(torch.fft.rfft(fr * self.win_a), n=self.L) * self.win_s; T = f.shape[-2]; length = (T - 1) * self.H + self.L
        raw = F.fold(f.reshape(B * C, T, self.L).transpose(1, 2), (1, length), (1, self.L), stride=(1, self.H)).reshape(B, C, length)
        den = F.fold(self.win_p[None, :, None].expand(1, -1, T), (1, length), (1, self.L), stride=(1, self.H)).reshape(1, 1, length)
        return (raw / den.clamp_min(1e-8))[:, :, self.left:self.left + N], den[0, 0, self.left:self.left + N]

    def init_state(self, B=1):
        z = self.win_a.new_zeros
        return [z(B, 2, self.left), None, z(B, 2, self.left), z(1, 1, self.left)]

    def step(self, x, state):
        history, h, tail, den = state
        fr = torch.cat([history, x], -1); z = torch.fft.rfft(fr * self.win_a).unsqueeze(2); y, h = self.spectral(z, h)
        raw = torch.fft.irfft(y[:, :, 0], n=self.L) * self.win_s; env = self.win_p[None, None, :]
        raw = raw + F.pad(tail, (0, self.H)); env = env + F.pad(den, (0, self.H))
        out = raw[:, :, :self.H] / env[:, :, :self.H].clamp_min(1e-8)
        return out, [fr[:, :, -self.left:], h, raw[:, :, self.H:], env[:, :, self.H:]]


class AsymStep(nn.Module):
    """ONNX-exportable fixed-512 block step (4 hops), real-valued DFT, same math as fast_music_ort.Step but with
    separate analysis / synthesis windows and the w_a*w_s envelope."""

    def __init__(self, m):
        super().__init__(); self.m = m; self.K = sum((b - a + s - 1) // s for a, b, s in m.regions)
        n = torch.arange(m.L, dtype=torch.float64); k = torch.arange(m.L // 2 + 1, dtype=torch.float64)
        a = 2 * math.pi * n[:, None] * k[None, :] / m.L
        self.register_buffer('co', a.cos().float()); self.register_buffer('si', a.sin().float())
        scale = torch.full((m.L // 2 + 1,), 2 / m.L); scale[0] = scale[-1] = 1 / m.L
        self.register_buffer('ico', (self.co * scale).T); self.register_buffer('isi', (self.si * scale).T)

    def initial(self):
        m = self.m
        return [torch.zeros(1, 2, m.left), torch.zeros(2, self.K, 64), torch.zeros(2, 1, 192), torch.zeros(1, 2, m.left), torch.zeros(1, 1, m.left)]

    def forward(self, block, hist, lh, gh, tail, den):
        m = self.m; ys = []
        for j in range(512 // m.H):
            fr = torch.cat([hist, block[:, :, j * m.H:(j + 1) * m.H]], -1); f = fr * m.win_a
            re = f @ self.co; im = -(f @ self.si); v = torch.cat([re, im], 1); v = v / (v.square().mean((1, 2), keepdim=True) + 1e-8).sqrt()
            pieces = []; sizes = []
            for (a, b, s), enc in zip(m.regions, m.enc):
                q = enc(F.pad(v[:, :, a:b], (0, (-(b - a)) % s))); pieces.append(q); sizes.append(q.shape[-1])
            q = F.silu(m.cross(torch.cat(pieces, -1)))
            pooled = torch.cat([q[:, :, i * self.K // 8:math.ceil((i + 1) * self.K / 8)].mean(-1, keepdim=True) for i in range(8)], -1)
            g = F.silu(m.global_in(pooled.flatten(1))).reshape(1, 1, 192); g, gh = m.global_rnn(g, gh)
            q = q + m.global_out(g).reshape(1, 64, 1); q = q.permute(2, 0, 1); q, lh = m.rnn(q, lh); q = q.permute(1, 2, 0)
            pred = []; off = 0
            for (a, b, s), dec, n in zip(m.regions, m.dec, sizes): pred.append(dec(q[:, :, off:off + n])[:, :, :b - a]); off += n
            mask = torch.tanh(torch.cat(pred, -1) + m.skip(v.transpose(1, 2)).transpose(1, 2))
            yr = re * mask[:, :2] - im * mask[:, 2:]; yi = re * mask[:, 2:] + im * mask[:, :2]
            raw = (yr @ self.ico - yi @ self.isi) * m.win_s + F.pad(tail, (0, m.H))
            env = m.win_p[None, None, :] + F.pad(den, (0, m.H))
            ys.append(raw[:, :, :m.H] / env[:, :, :m.H].clamp_min(1e-8))
            hist = fr[:, :, -m.left:]; tail = raw[:, :, m.H:]; den = env[:, :, m.H:]
        return torch.cat(ys, -1), hist, lh, gh, tail, den


def make(m, path):
    import onnxruntime as ort
    core = AsymStep(m).eval(); initial = core.initial(); names = ['block', 'hist', 'lh', 'gh', 'tail', 'den']
    torch.onnx.export(core, (torch.zeros(1, 2, 512), *initial), str(path), input_names=names, opset_version=17, dynamo=False)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])

    def step(b, st):
        y = sess.run(None, dict(zip(names, [b[None].numpy(), *st]))); return torch.from_numpy(y[0][0]), y[1:]
    return step, lambda: [t.numpy().copy() for t in initial]
