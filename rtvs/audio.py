import numpy as np
from . import io as rio
from .common import STEMS, track_dir


def load_track(name, split, dtype="float32"):
    """Returns dict of (T, 2) arrays for mixture + 4 stems, and sr."""
    d = track_dir(name, split)
    out, sr0 = {}, None
    for k in ["mixture"] + STEMS:
        x, sr = rio.read(f"{d}/{k}.wav", dtype=dtype, always_2d=True)
        sr0 = sr0 or sr
        assert sr == sr0
        out[k] = x
    return out, sr0
