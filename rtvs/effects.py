"""Random mixing-effects chain used for the robustness evaluation (and for the effects fine-tuning experiment).

Per stem: 1-3 RBJ peaking EQs (60 Hz-16 kHz, +-6 dB, Q 0.5-2); with p=0.5 a feed-forward compressor
(ratio 2-8, threshold -30..-10 dB, attack 1-30 ms, release 50-300 ms); vocals and "other" get, with p=0.5,
a synthetic exponential-decay reverb (wet 0.1-0.4, RT 0.2-0.6 s). With p=0.5 the summed mix then goes through
an instantaneous, stereo-linked peak limiter (threshold -6..-1 dBFS) whose gain is applied to every stem, so
the mixture stays the exact sum of the processed stems and the processed vocals stay a valid target.
Everything is deterministic given ``seed``. Arrays are (channels, samples)."""
import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, lfilter

SR = 44100
_SRC = Path(__file__).with_name("_envelope.c")
_LIB = None


def _load_lib():
    """Compile the tiny C envelope follower once (cached); return None if no C compiler is available."""
    global _LIB
    if _LIB is None:
        cache = Path(os.environ.get("RTVS_CACHE", Path(tempfile.gettempdir()) / "rtvs_cache"))
        so = cache / "envelope.so"
        try:
            if not so.exists():
                cache.mkdir(parents=True, exist_ok=True)
                subprocess.run([os.environ.get("CC", "cc"), "-O2", "-shared", "-fPIC", str(_SRC), "-o", str(so)],
                               check=True, capture_output=True)
            _LIB = ctypes.CDLL(str(so))
            _LIB.envelope.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_double, ctypes.c_double]
        except Exception:
            _LIB = False
    return _LIB or None


def envelope(x, attack, release):
    """One-pole attack/release peak follower (same recursion as _envelope.c)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.empty_like(x)
    a_att, a_rel = np.exp(-1 / (attack * SR)), np.exp(-1 / (release * SR))
    lib = _load_lib()
    if lib is not None:
        lib.envelope(x.ctypes.data, y.ctypes.data, len(x), a_att, a_rel)
        return y
    state = 0.0                                    # pure-Python fallback (slow on whole tracks)
    for i, v in enumerate(x):
        a = a_att if v > state else a_rel
        state = a * state + (1.0 - a) * v
        y[i] = state
    return y


def transform(stems, seed):
    """stems: (4, 2, T) in order vocals, drums, bass, other. Returns (processed stems float32, parameters)."""
    rng = np.random.default_rng(seed)
    out, params = [], []
    for idx, stem in enumerate(stems):
        x = np.array(stem, dtype=np.float64, copy=True)
        record = {"eq": []}
        for _ in range(int(rng.integers(1, 4))):
            hz = float(np.exp(rng.uniform(np.log(60), np.log(16000))))
            gain = float(rng.uniform(-6, 6))
            q = float(rng.uniform(.5, 2))
            A = 10 ** (gain / 40); w = 2 * np.pi * hz / SR; alpha = np.sin(w) / (2 * q)
            b = np.array([1 + alpha * A, -2 * np.cos(w), 1 - alpha * A])
            a = np.array([1 + alpha / A, -2 * np.cos(w), 1 - alpha / A])
            x = lfilter(b / a[0], a / a[0], x, axis=-1)
            record["eq"].append([hz, gain, q])
        if rng.random() < .5:
            ratio = float(rng.uniform(2, 8)); tdb = float(rng.uniform(-30, -10))
            attack = float(rng.uniform(.001, .030)); release = float(rng.uniform(.05, .3))
            env = envelope(np.max(np.abs(x), axis=0), attack, release)
            g = np.minimum(1, (np.maximum(env, 1e-12) / 10 ** (tdb / 20)) ** (1 / ratio - 1))
            x *= g
            record["compressor"] = [ratio, tdb, attack, release]
        if idx in (0, 3) and rng.random() < .5:
            wet = float(rng.uniform(.1, .4)); rt = float(rng.uniform(.2, .6)); n = int(rt * SR)
            ir = rng.normal(size=n) * np.exp(-np.log(1000) * np.arange(n) / (rt * SR))
            ir[:int(.01 * SR)] = 0
            ir /= max(np.linalg.norm(ir), 1e-12)
            x = (1 - wet) * x + wet * np.stack([fftconvolve(c, ir)[:x.shape[-1]] for c in x])
            record["reverb"] = [wet, rt]
        out.append(x)
        params.append(record)
    mix = sum(out)
    limiter = None
    if rng.random() < .5:
        threshold = float(10 ** (rng.uniform(-6, -1) / 20))
        g = np.minimum(1, threshold / np.maximum(np.max(np.abs(mix), axis=0), 1e-12))
        out = [s * g for s in out]
        limiter = {"threshold": threshold, "instantaneous_stereo_linked": True}
    out = np.stack(out).astype("float32")
    return out, {"stems": params, "limiter": limiter, "seed": int(seed)}


def effect_seed(split, track_index):
    """Seed rule used for the reported robustness test: one fixed chain per track."""
    return 20261111 + track_index + (0 if split == "valid" else 100000)
