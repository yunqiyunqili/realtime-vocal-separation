"""WAV reading with retries (useful on network file systems).
On a read failure, retry up to 3 times after 2 s, 5 s and 10 s, logging each retry to stderr. After the last
retry the original exception is re-raised: tracks are never skipped or replaced by silence."""
import datetime
import sys
import time

import soundfile as sf

RETRY_DELAYS = (2, 5, 10)


def _log(msg):
    print(f"[io-retry] {datetime.datetime.now().isoformat(timespec='seconds')} {msg}", file=sys.stderr, flush=True)


def _with_retry(fn, path, what):
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            out = fn()
            if attempt:
                _log(f"OK after {attempt} retr{'y' if attempt == 1 else 'ies'}: {what} {path!r}")
            return out
        except Exception as e:  # noqa: BLE001 - any I/O / decoder error is retried, then re-raised
            if attempt == len(RETRY_DELAYS):
                _log(f"giving up after {attempt} retries: {what} {path!r}: {type(e).__name__}: {e}")
                raise
            d = RETRY_DELAYS[attempt]
            _log(f"retry {attempt + 1}/{len(RETRY_DELAYS)} in {d}s: {what} {path!r}: {type(e).__name__}: {e}")
            time.sleep(d)


def read(path, **kw):
    """Same signature and semantics as soundfile.read."""
    return _with_retry(lambda: sf.read(path, **kw), path, "read")


def info(path):
    return _with_retry(lambda: sf.info(path), path, "info")
