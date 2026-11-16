"""Training data: random 6-s chunks from MUSDB18-HQ WAV stems.
Augmentation (per stem): random gain U[gain_lo, gain_hi], random L/R swap (p=0.5).
Cross-track mixing (prob mix_prob per item): each stem drawn from an independently chosen
training track and offset.  The mixture is ALWAYS re-summed from the augmented stems
(mixture.wav is never used for training).  No EQ / compression / reverb.
Deterministic: item i uses np.random.default_rng([seed, epoch, i])."""
import numpy as np, torch
from . import io as rio
from .common import STEMS, track_dir, SR


class MusdbChunks(torch.utils.data.Dataset):
    def __init__(self, tracks, split="train", chunk_s=6.0, items=1000, mix_prob=0.5,
                 gain=(0.25, 1.25), swap_p=0.5, seed=0):
        assert split in ("train", "valid")  # never test
        self.tracks, self.split = list(tracks), split
        self.T = int(round(chunk_s * SR)); self.items = items
        self.mix_prob, self.gain, self.swap_p, self.seed, self.epoch = mix_prob, gain, swap_p, seed, 0
        self.len = {t: rio.info(f"{track_dir(t, split)}/vocals.wav").frames for t in self.tracks}

    def __len__(self):
        return self.items

    def _read(self, t, stem, start):
        x, _ = rio.read(f"{track_dir(t, self.split)}/{stem}.wav", start=start, frames=self.T,
                       dtype="float32", always_2d=True)
        if len(x) < self.T:
            x = np.pad(x, ((0, self.T - len(x)), (0, 0)))
        return x.T

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        main = self.tracks[rng.integers(len(self.tracks))]
        cross = rng.random() < self.mix_prob
        start_main = int(rng.integers(0, max(1, self.len[main] - self.T)))
        stems = {}
        for s in STEMS:
            t = self.tracks[rng.integers(len(self.tracks))] if cross else main
            st = int(rng.integers(0, max(1, self.len[t] - self.T))) if cross else start_main
            x = self._read(t, s, st) * rng.uniform(*self.gain)
            if rng.random() < self.swap_p:
                x = x[::-1]
            stems[s] = np.ascontiguousarray(x)
        mix = sum(stems.values())
        return torch.from_numpy(mix.astype(np.float32)), torch.from_numpy(stems["vocals"].astype(np.float32))


class MusdbChunksMulti(MusdbChunks):
    """Four-target variant (used by the RT-STT experiment): same sampling / augmentation as MusdbChunks (random chunk, per-stem gain and L/R swap,
    cross-track mixing with prob mix_prob, mixture re-summed from augmented stems; no pitch/tempo), but returns
    ALL four stems as targets: (mix (2, T), stems (4, 2, T)) in order vocals, drums, bass, other."""

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        main = self.tracks[rng.integers(len(self.tracks))]
        cross = rng.random() < self.mix_prob
        start_main = int(rng.integers(0, max(1, self.len[main] - self.T)))
        stems = []
        for s in STEMS:
            t = self.tracks[rng.integers(len(self.tracks))] if cross else main
            st = int(rng.integers(0, max(1, self.len[t] - self.T))) if cross else start_main
            x = self._read(t, s, st) * rng.uniform(*self.gain)
            if rng.random() < self.swap_p:
                x = x[::-1]
            stems.append(np.ascontiguousarray(x))
        stems = np.stack(stems).astype(np.float32)
        return torch.from_numpy(stems.sum(0)), torch.from_numpy(stems)
