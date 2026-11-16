"""Shared constants and helpers.

The MUSDB18-HQ location is never hard-coded. Set it with the environment variable ``MUSDB18HQ_ROOT`` or pass
``--musdb-root`` to any script (see ``add_data_args``). The directory must contain ``train/`` and ``test/``
sub-directories with one folder per track (``mixture.wav``, ``vocals.wav``, ``drums.wav``, ``bass.wav``,
``other.wav``), i.e. the standard MUSDB18-HQ layout.
"""
import datetime
import json
import os
import platform
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 44100
STEMS = ["vocals", "drums", "bass", "other"]
_MUSDB_ROOT = os.environ.get("MUSDB18HQ_ROOT")


def set_musdb_root(path):
    global _MUSDB_ROOT
    if path:
        _MUSDB_ROOT = os.path.abspath(os.path.expanduser(path))


def musdb_root():
    if not _MUSDB_ROOT:
        raise RuntimeError("MUSDB18-HQ location unknown: set MUSDB18HQ_ROOT or pass --musdb-root")
    return _MUSDB_ROOT


def add_data_args(ap):
    ap.add_argument("--musdb-root", default=None,
                    help="MUSDB18-HQ root (contains train/ and test/); default: $MUSDB18HQ_ROOT")
    return ap


def apply_data_args(args):
    set_musdb_root(getattr(args, "musdb_root", None))


def read_split(name):
    """Track names of a split. valid = the official 14-track MUSDB18 validation subset of train/."""
    with open(os.path.join(REPO, "splits", f"{name}.txt")) as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def track_dir(name, split):
    sub = "test" if split == "test" else "train"
    return os.path.join(musdb_root(), sub, name)


def provenance(extra=None):
    """Git commit, dirty flag and library versions, stored next to every result."""
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, cwd=REPO, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    d = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(sh(["git", "status", "--porcelain", "--untracked-files=no"])),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    try:
        import torch
        d["torch"] = torch.__version__
        d["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            d["gpu"] = torch.cuda.get_device_name(0)
        d["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    if extra:
        d.update(extra)
    return d


def dump_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, path)
