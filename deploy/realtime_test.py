#!/usr/bin/env python3
"""Real-time test of the F12 causal Conv-TasNet vocal separator on a real sound card.

Mode A ("file"):    wav file -> model -> speakers. Measures per-block inference time, blocks over the 11.61 ms
                    budget, and output underflows (xruns) reported by the audio driver.
Mode B ("latency"): full-duplex click test. Plays clicks and records them back with the microphone (or a cable
                    loopback), once directly ("passthrough") and once through the model, and estimates the
                    round-trip delay of each by cross-correlation.

Dependencies: numpy, soundfile, sounddevice, onnxruntime. Python 3.9+. Windows / macOS / Linux.
Run `python realtime_test.py --help` and see README.md.
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import threading
import time

import numpy as np

SR = 44100
BLOCK = 512
BUDGET_MS = BLOCK / SR * 1000.0          # 11.61 ms
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "f12_stream_block512.onnx")
DEFAULT_WAV = os.path.join(HERE, "clip.wav")
OUTPUTS = ("vocals", "mixture", "accompaniment")


# --------------------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------------------
class Separator:
    """Stateful block-512 streaming separator. process() takes (2, 512) float32 and returns
    (vocals, aligned_mixture), both (2, 512): the model output is delayed by `delay` samples relative to its input,
    so the mixture is passed through a delay line of the same length to stay sample-aligned."""

    def __init__(self, model_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if not os.path.exists(model_path):
            raise SystemExit("model not found: %s\nDownload f12_stream_block512.onnx (+ model_meta.json) from the GitHub "
                             "Release into deploy/, or export it with scripts/export_onnx.py." % model_path)
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inputs = self.sess.get_inputs()
        self.names = [i.name for i in self.inputs]
        shape = self.inputs[0].shape
        if list(shape) != [2, BLOCK]:
            raise SystemExit("unexpected model input shape %s (expected [2, %d])" % (shape, BLOCK))
        meta_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "model_meta.json")
        self.delay = 384
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.delay = int(json.load(f)["output_delay_samples"])
        self.reset()

    def reset(self):
        self.state = [np.zeros(i.shape, np.float32) for i in self.inputs[1:]]
        self.hist = np.zeros((2, self.delay), np.float32)

    def process(self, block):
        feed = {self.names[0]: np.ascontiguousarray(block, dtype=np.float32)}
        for n, s in zip(self.names[1:], self.state):
            feed[n] = s
        t0 = time.perf_counter()
        r = self.sess.run(None, feed)
        dt = (time.perf_counter() - t0) * 1000.0
        self.state = r[1:]
        buf = np.concatenate([self.hist, block], axis=1)
        mix_aligned = buf[:, :BLOCK]
        self.hist = buf[:, BLOCK:]
        return r[0], mix_aligned, dt


# --------------------------------------------------------------------------------------------------------------
# System information
# --------------------------------------------------------------------------------------------------------------
def cpu_model():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if out:
                return out
        elif sys.platform.startswith("win"):
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def system_info():
    info = {"cpu": cpu_model(), "logical_cpus": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__}
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except Exception:
        pass
    try:
        import sounddevice as sd
        info["sounddevice"] = sd.__version__
        info["portaudio"] = sd.get_portaudio_version()[1]
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------------------------------------------
def resolve_device(sd, spec, hostapi, kind, channels):
    """spec: None (default device), an integer index, or a case-insensitive substring of the device name.
    hostapi: optional substring of the host API name (e.g. 'WASAPI', 'ASIO', 'Core Audio', 'ALSA')."""
    key = "max_output_channels" if kind == "output" else "max_input_channels"
    if spec is not None and str(spec).isdigit():
        return int(spec)
    if spec is None and hostapi is None:
        return None
    apis = sd.query_hostapis()
    cands = []
    for idx, d in enumerate(sd.query_devices()):
        api = apis[d["hostapi"]]["name"]
        if d[key] < channels:
            continue
        if hostapi and hostapi.lower() not in api.lower():
            continue
        if spec and str(spec).lower() not in d["name"].lower():
            continue
        cands.append(idx)
    if not cands:
        raise SystemExit("no %s device matches --device %r --hostapi %r with >= %d channels; run --list-devices"
                         % (kind, spec, hostapi, channels))
    if spec is None:                                   # host API only: use its default device of this kind
        for a in apis:
            if hostapi.lower() in a["name"].lower():
                dflt = a["default_output_device" if kind == "output" else "default_input_device"]
                if dflt in cands:
                    return dflt
    return cands[0]


def device_desc(sd, idx, kind):
    if idx is None:
        idx = sd.default.device[1 if kind == "output" else 0]
    d = sd.query_devices(idx)
    return {"index": int(idx), "name": d["name"], "hostapi": sd.query_hostapis(d["hostapi"])["name"],
            "default_samplerate": d["default_samplerate"]}


def extra_settings(sd, args, dev_idx):
    if not args.wasapi_exclusive:
        return None
    if dev_idx is None or "wasapi" not in sd.query_hostapis(sd.query_devices(dev_idx)["hostapi"])["name"].lower():
        raise SystemExit("--wasapi-exclusive needs a WASAPI device (use --hostapi WASAPI)")
    return sd.WasapiSettings(exclusive=True)


# --------------------------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------------------------
class Stats:
    FLAGS = ("output_underflow", "output_overflow", "input_underflow", "input_overflow", "priming_output")

    def __init__(self, n_max):
        self.inf = np.zeros(n_max, np.float64)
        self.cb = np.zeros(n_max, np.float64)
        self.n = 0
        self.flags = dict((f, 0) for f in self.FLAGS)
        self.odd_frames = 0

    def status(self, status):
        if status:
            for f in self.FLAGS:
                if getattr(status, f, False):
                    self.flags[f] += 1

    def add(self, inf_ms, cb_ms):
        if self.n < len(self.inf):
            self.inf[self.n] = inf_ms
            self.cb[self.n] = cb_ms
            self.n += 1

    def summary(self, skip=10):
        t = self.inf[skip:self.n] if self.n > skip + 10 else self.inf[:self.n]
        c = self.cb[skip:self.n] if self.n > skip + 10 else self.cb[:self.n]
        if len(t) == 0:
            return {"blocks": 0}
        return {"blocks": int(self.n), "blocks_in_stats": int(len(t)), "warmup_blocks_excluded": int(self.n - len(t)),
                "budget_ms": BUDGET_MS,
                "inference_ms": {"p50": float(np.median(t)), "p99": float(np.percentile(t, 99)), "max": float(t.max()),
                                 "mean": float(t.mean())},
                "callback_ms": {"p50": float(np.median(c)), "p99": float(np.percentile(c, 99)), "max": float(c.max())},
                "inference_over_budget": int((t > BUDGET_MS).sum()),
                "inference_over_budget_ratio": float((t > BUDGET_MS).mean()),
                "callback_over_budget": int((c > BUDGET_MS).sum()),
                "rtf_median": float(np.median(t) / BUDGET_MS),
                "xruns_output_underflow": int(self.flags["output_underflow"]),
                "status_flags": dict(self.flags), "callbacks_with_unexpected_frame_count": int(self.odd_frames)}


# --------------------------------------------------------------------------------------------------------------
# Mode A: file -> model -> speakers
# --------------------------------------------------------------------------------------------------------------
class FilePlayer:
    def __init__(self, sep, audio, output, n_max):
        self.sep, self.audio, self.output = sep, audio, output     # audio: (T, 2) float32
        self.pos = 0
        self.stats = Stats(n_max)

    def next_block(self):
        T = len(self.audio)
        idx = (self.pos + np.arange(BLOCK)) % T                   # loop the file
        self.pos = (self.pos + BLOCK) % T
        return self.audio[idx].T

    def callback(self, outdata, frames, time_info, status):
        t0 = time.perf_counter()
        self.stats.status(status)
        if frames != BLOCK:
            self.stats.odd_frames += 1
            outdata.fill(0)
            return
        voc, mix, inf_ms = self.sep.process(self.next_block())
        o = self.output
        y = voc if o == "vocals" else (mix if o == "mixture" else mix - voc)
        np.clip(y.T, -1.0, 1.0, out=outdata)
        self.stats.add(inf_ms, (time.perf_counter() - t0) * 1000.0)


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit("%s is %d Hz; the model needs %d Hz. Resample it first (e.g. ffmpeg -i in.wav -ar 44100 out.wav)."
                         % (path, sr, SR))
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return np.ascontiguousarray(x)


def keyboard_thread(player, stop):
    print("  keys + Enter:  v = vocals   m = mixture   a = accompaniment   q = quit", flush=True)
    while not stop.is_set():
        try:
            s = sys.stdin.readline()
        except Exception:
            return
        if not s:
            return
        s = s.strip().lower()
        if s in ("v", "m", "a"):
            player.output = {"v": "vocals", "m": "mixture", "a": "accompaniment"}[s]
            print("  -> output: %s" % player.output, flush=True)
        elif s == "q":
            stop.set()


def run_file(args, sep, result):
    if os.path.exists(args.wav):
        audio = load_wav(args.wav)
    elif args.simulate:
        print("no wav at %s: simulating with 30 s of low-level noise" % args.wav, flush=True)
        audio = (0.05 * np.random.default_rng(0).standard_normal((30 * SR, 2))).astype(np.float32)
    else:
        raise SystemExit("wav not found: %s\nPass any 44.1 kHz music file with --wav (no audio is shipped with the "
                         "repository; MUSDB18-HQ tracks may not be redistributed)." % args.wav)
    n_max = int(args.duration * SR / BLOCK) + 1000
    player = FilePlayer(sep, audio, args.output, n_max)
    result["mode"] = "file"
    result["settings"].update({"wav": os.path.basename(args.wav), "output": args.output, "duration_s": args.duration})
    if args.simulate:
        return simulate_file(args, player, result)
    import sounddevice as sd
    dev = resolve_device(sd, args.device, args.hostapi, "output", 2)
    try:
        sd.check_output_settings(device=dev, samplerate=SR, channels=2, dtype="float32",
                                 extra_settings=extra_settings(sd, args, dev))
    except Exception as e:
        raise SystemExit("output device does not accept 44100 Hz stereo float32: %s\n"
                         "See README 'sample rate mismatch'." % e)
    stop = threading.Event()
    stream = sd.OutputStream(samplerate=SR, blocksize=BLOCK, channels=2, dtype="float32", device=dev,
                             latency=args.latency, callback=player.callback,
                             extra_settings=extra_settings(sd, args, dev))
    result["device"] = {"output": device_desc(sd, dev, "output")}
    print("Mode A: %s -> model -> %s (%s), %d s. Output: %s" % (os.path.basename(args.wav), result["device"]["output"]["name"],
          result["device"]["output"]["hostapi"], args.duration, args.output), flush=True)
    if not args.no_keys:
        threading.Thread(target=keyboard_thread, args=(player, stop), daemon=True).start()
    t_start = time.perf_counter()
    with stream:
        result["stream"] = {"samplerate": stream.samplerate, "blocksize": stream.blocksize,
                            "latency_output_s": stream.latency}
        stop.wait(args.duration)
    result["wall_s"] = time.perf_counter() - t_start
    result["stats"] = player.stats.summary()
    return result


def simulate_file(args, player, result):
    """No audio device: call the callback directly n times (as fast as possible), same code path."""
    class _Status:
        output_underflow = False

        def __bool__(self):
            return False
    out = np.zeros((BLOCK, 2), np.float32)
    n = args.simulate
    t0 = time.perf_counter()
    for _ in range(n):
        player.callback(out, BLOCK, None, _Status())
    result["simulated"] = True
    result["device"] = {"output": "simulated (no device opened)"}
    result["stream"] = {"samplerate": SR, "blocksize": BLOCK}
    result["wall_s"] = time.perf_counter() - t0
    result["stats"] = player.stats.summary()
    result["last_block_peak"] = float(np.abs(out).max())
    return result


# --------------------------------------------------------------------------------------------------------------
# Mode B: click round-trip latency (passthrough vs through the model), in ONE full-duplex stream
# --------------------------------------------------------------------------------------------------------------
def click_template():
    """20 ms Hann-windowed exponential sweep 300 Hz - 8 kHz: click-like but with a sharp autocorrelation peak."""
    n = int(0.02 * SR)
    t = np.arange(n) / SR
    f0, f1, T = 300.0, 8000.0, n / SR
    k = np.log(f1 / f0)
    phase = 2 * np.pi * f0 * T / k * (np.exp(t / T * k) - 1)
    return (np.sin(phase) * np.hanning(n)).astype(np.float32)


class ClickTest:
    """Odd slots: click written straight to the output (passthrough). Even slots: click fed into the model; the output
    is the model's accompaniment (aligned mixture - vocals), which carries the click with the model's full delay.
    Recording and both references share the stream's frame counter, so lag = round trip."""

    def __init__(self, sep, n_clicks, gap_s, amp, in_ch):
        self.sep, self.amp, self.in_ch = sep, amp, in_ch
        self.tpl = click_template()
        self.gap = int(gap_s * SR)
        self.total = int((0.5 + 2 * n_clicks * gap_s + 1.0) * SR)
        self.total += (-self.total) % BLOCK
        self.ref_pass = np.zeros(self.total, np.float32)
        self.ref_model = np.zeros(self.total, np.float32)
        self.t_pass, self.t_model = [], []
        t = int(0.5 * SR)
        for _ in range(n_clicks):
            self.ref_pass[t:t + len(self.tpl)] = self.tpl * amp; self.t_pass.append(t); t += self.gap
            self.ref_model[t:t + len(self.tpl)] = self.tpl * amp; self.t_model.append(t); t += self.gap
        self.rec = np.zeros(self.total, np.float32)
        self.pos = 0
        self.stats = Stats(self.total // BLOCK + 10)
        self.done = threading.Event()

    def callback(self, indata, outdata, frames, time_info, status):
        t0 = time.perf_counter()
        self.stats.status(status)
        p = self.pos
        if frames != BLOCK or p + BLOCK > self.total:
            outdata.fill(0)
            if p + BLOCK > self.total:
                self.done.set()
            else:
                self.stats.odd_frames += 1
            return
        self.rec[p:p + BLOCK] = indata[:, :self.in_ch].mean(axis=1)
        mb = np.repeat(self.ref_model[None, p:p + BLOCK], 2, axis=0)
        voc, mix, inf_ms = self.sep.process(mb)
        y = (mix - voc) + self.ref_pass[None, p:p + BLOCK]
        np.clip(y.T, -1.0, 1.0, out=outdata)
        self.pos = p + BLOCK
        self.stats.add(inf_ms, (time.perf_counter() - t0) * 1000.0)

    def estimate(self, starts, max_lag_s=0.4):
        """Per click: cross-correlate the recording after the click with the template (FFT), lag of the |peak|."""
        L = int(max_lag_s * SR)
        tpl = self.tpl
        out = []
        for s in starts:
            seg = self.rec[s:s + L + len(tpl)]
            if len(seg) < len(tpl) + 1:
                continue
            n = 1 << int(np.ceil(np.log2(len(seg) + len(tpl))))
            xc = np.fft.irfft(np.fft.rfft(seg, n) * np.conj(np.fft.rfft(tpl, n)), n)[:len(seg) - len(tpl) + 1]
            a = np.abs(xc)
            lag = int(a.argmax())
            noise = float(np.median(a)) + 1e-12
            out.append({"lag_samples": lag, "lag_ms": lag / SR * 1000.0, "peak_to_median": float(a[lag] / noise)})
        return out


def summarize_lags(lags, min_conf):
    good = [x for x in lags if x["peak_to_median"] >= min_conf]
    if not good:
        return {"n_clicks": len(lags), "n_reliable": 0, "per_click": lags}
    ms = np.array([x["lag_ms"] for x in good])
    return {"n_clicks": len(lags), "n_reliable": len(good), "median_ms": float(np.median(ms)), "min_ms": float(ms.min()),
            "max_ms": float(ms.max()), "spread_ms": float(ms.max() - ms.min()), "per_click": lags}


def run_latency(args, sep, result):
    result["mode"] = "latency"
    ct = ClickTest(sep, args.clicks, args.click_gap, args.click_amp, 1)
    result["settings"].update({"clicks_per_path": args.clicks, "click_gap_s": args.click_gap, "click_amp": args.click_amp,
                               "click": "20 ms Hann-windowed exponential sweep 300 Hz-8 kHz",
                               "model_path_signal": "accompaniment output (aligned mixture - vocals)"})
    if args.simulate:
        simulate_loopback(args, ct, result)
    else:
        import sounddevice as sd
        odev = resolve_device(sd, args.device, args.hostapi, "output", 2)
        idev = resolve_device(sd, args.input_device if args.input_device is not None else args.device, args.hostapi, "input", 1)
        in_ch = 1
        ct.in_ch = in_ch
        try:
            sd.check_output_settings(device=odev, samplerate=SR, channels=2, dtype="float32")
            sd.check_input_settings(device=idev, samplerate=SR, channels=in_ch, dtype="float32")
        except Exception as e:
            raise SystemExit("device does not accept 44100 Hz float32: %s\nSee README 'sample rate mismatch'." % e)
        ex = extra_settings(sd, args, odev)
        stream = sd.Stream(samplerate=SR, blocksize=BLOCK, channels=(in_ch, 2), dtype="float32", device=(idev, odev),
                           latency=args.latency, callback=ct.callback, extra_settings=(ex, ex) if ex else None)
        result["device"] = {"input": device_desc(sd, idev, "input"), "output": device_desc(sd, odev, "output")}
        print("Mode B: clicks on %s -> recorded by %s, %d clicks per path, %.1f s"
              % (result["device"]["output"]["name"], result["device"]["input"]["name"], args.clicks, ct.total / SR), flush=True)
        print("  keep the room quiet; for a cable loopback connect line-out -> line-in", flush=True)
        with stream:
            result["stream"] = {"samplerate": stream.samplerate, "blocksize": stream.blocksize,
                                "latency_input_s": stream.latency[0], "latency_output_s": stream.latency[1],
                                "reported_roundtrip_ms": (stream.latency[0] + stream.latency[1]) * 1000.0}
            ct.done.wait(ct.total / SR + 5.0)
    rec_peak = float(np.abs(ct.rec).max())
    lp = summarize_lags(ct.estimate(ct.t_pass), args.min_confidence)
    lm = summarize_lags(ct.estimate(ct.t_model), args.min_confidence)
    res = {"recording_peak": rec_peak, "passthrough": lp, "through_model": lm,
           "model_output_delay_samples": sep.delay, "model_output_delay_ms": sep.delay / SR * 1000.0}
    if lp.get("n_reliable") and lm.get("n_reliable"):
        res["model_extra_ms"] = lm["median_ms"] - lp["median_ms"]
    result["latency"] = res
    result["stats"] = ct.stats.summary(skip=0)
    return result


def simulate_loopback(args, ct, result):
    """No device: output is fed back to the input after a fixed delay (+ noise), so the expected answer is known."""
    D = args.sim_loop_delay
    played = np.zeros(ct.total + BLOCK, np.float32)           # output timeline (left channel)
    rng = np.random.default_rng(0)
    out = np.zeros((BLOCK, 2), np.float32)

    class _S:
        def __bool__(self):
            return False
    k = 0
    while not ct.done.is_set():
        t = k * BLOCK                                          # input at time t = 0.3 * output at time t - D, + noise
        src = np.zeros(BLOCK, np.float32)
        lo, hi = max(t - D, 0), max(t - D + BLOCK, 0)
        if hi > lo:
            src[BLOCK - (hi - lo):] = played[lo:hi]
        inp = (0.3 * src + rng.normal(0, 1e-3, BLOCK)).astype(np.float32)[:, None]
        ct.callback(inp, out, BLOCK, None, _S())
        if t + BLOCK <= len(played):
            played[t:t + BLOCK] = out[:, 0]
        k += 1
    result["simulated"] = True
    result["device"] = {"input": "simulated loopback", "output": "simulated loopback"}
    result["stream"] = {"samplerate": SR, "blocksize": BLOCK, "simulated_loop_delay_samples": D,
                        "expected_passthrough_ms": D / SR * 1000.0,
                        "expected_through_model_ms": (D + ct.sep.delay) / SR * 1000.0}


# --------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------
def human_summary(r):
    L = []
    s = r.get("system", {})
    L.append("=" * 72)
    L.append("F12 real-time test (%s%s)" % (r["mode"], ", SIMULATED - no audio device" if r.get("simulated") else ""))
    L.append("  CPU: %s (%s logical)   OS: %s" % (s.get("cpu"), s.get("logical_cpus"), s.get("os")))
    L.append("  onnxruntime %s, threads 1/1, sample rate %d Hz, blocksize %d (budget %.2f ms)"
             % (s.get("onnxruntime"), SR, BLOCK, BUDGET_MS))
    dev = r.get("device", {})
    for k in ("output", "input"):
        if k in dev:
            d = dev[k]
            L.append("  %s device: %s" % (k, d if isinstance(d, str) else "%s [%s] (#%d)" % (d["name"], d["hostapi"], d["index"])))
    st = r.get("stream", {})
    if "latency_output_s" in st:
        L.append("  stream latency reported by sounddevice: output %.1f ms%s" % (
            st["latency_output_s"] * 1000, (", input %.1f ms" % (st["latency_input_s"] * 1000)) if "latency_input_s" in st else ""))
    x = r.get("stats", {})
    if x.get("blocks"):
        i = x["inference_ms"]
        L.append("  blocks: %d (%d in stats)" % (x["blocks"], x["blocks_in_stats"]))
        L.append("  inference ms: p50 %.2f  p99 %.2f  max %.2f   RTF %.2f" % (i["p50"], i["p99"], i["max"], x["rtf_median"]))
        L.append("  blocks over %.2f ms budget: %d (%.3f%%)   callback p99 %.2f ms"
                 % (BUDGET_MS, x["inference_over_budget"], 100 * x["inference_over_budget_ratio"], x["callback_ms"]["p99"]))
        L.append("  xruns (output_underflow): %d   other flags: %s"
                 % (x["xruns_output_underflow"], {k: v for k, v in x["status_flags"].items() if v and k != "output_underflow"} or "none"))
    lt = r.get("latency")
    if lt:
        for name, key in (("passthrough", "passthrough"), ("through model", "through_model")):
            v = lt[key]
            if v.get("n_reliable"):
                L.append("  round trip, %-13s: median %.1f ms (min %.1f, max %.1f; %d/%d clicks reliable)"
                         % (name, v["median_ms"], v["min_ms"], v["max_ms"], v["n_reliable"], v["n_clicks"]))
            else:
                L.append("  round trip, %-13s: NO reliable click detected (recording peak %.4f) - check mic / volume"
                         % (name, lt["recording_peak"]))
        if "model_extra_ms" in lt:
            L.append("  model adds %.1f ms (expected model output delay %d samples = %.1f ms)"
                     % (lt["model_extra_ms"], lt["model_output_delay_samples"], lt["model_output_delay_ms"]))
        if "reported_roundtrip_ms" in st:
            L.append("  (sounddevice reported input+output latency: %.1f ms)" % st["reported_roundtrip_ms"])
    L.append("=" * 72)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["file", "latency"], default="file", help="A = file (default), B = latency")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--wav", default=DEFAULT_WAV, help="44.1 kHz wav for mode A (mono is duplicated)")
    ap.add_argument("--output", choices=OUTPUTS, default="vocals", help="what mode A plays (switch live with v/m/a + Enter)")
    ap.add_argument("--duration", type=float, default=120.0, help="mode A run time in seconds (file loops)")
    ap.add_argument("--device", default=None, help="output device index or name substring (mode B: also input if --input-device not set)")
    ap.add_argument("--input-device", default=None, help="mode B input device index or name substring")
    ap.add_argument("--hostapi", default=None, help="host API substring, e.g. WASAPI, ASIO, 'Core Audio', ALSA")
    ap.add_argument("--wasapi-exclusive", action="store_true", help="Windows: open the WASAPI device in exclusive mode")
    ap.add_argument("--latency", default="low", help="'low', 'high' or seconds; PortAudio suggested latency")
    ap.add_argument("--clicks", type=int, default=8, help="mode B clicks per path")
    ap.add_argument("--click-gap", type=float, default=0.5, help="mode B seconds between clicks")
    ap.add_argument("--click-amp", type=float, default=0.5, help="mode B click amplitude (0-1)")
    ap.add_argument("--min-confidence", type=float, default=8.0, help="mode B: min correlation peak/median for a click to count")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--no-keys", action="store_true", help="mode A: do not read the keyboard")
    ap.add_argument("--json", default=None, help="result file (default: results_<mode>_<time>.json next to this script)")
    ap.add_argument("--simulate", type=int, default=0, metavar="N",
                    help="smoke test without a sound card: mode A runs N callbacks; mode B uses a simulated loopback")
    ap.add_argument("--sim-loop-delay", type=int, default=3000, help=argparse.SUPPRESS)
    args = ap.parse_args()
    try:
        args.latency = float(args.latency)
    except ValueError:
        pass

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        print("\nhost APIs:", ", ".join(a["name"] for a in sd.query_hostapis()))
        return

    sep = Separator(args.model)
    result = {"tool": "realtime_test.py", "time": datetime.datetime.now().isoformat(timespec="seconds"),
              "system": system_info(), "model": os.path.basename(args.model), "model_output_delay_samples": sep.delay,
              "settings": {"samplerate": SR, "blocksize": BLOCK, "budget_ms": BUDGET_MS,
                           "onnxruntime_threads": {"intra_op": 1, "inter_op": 1}, "suggested_latency": args.latency}}
    if args.mode == "file":
        run_file(args, sep, result)
    else:
        run_latency(args, sep, result)

    path = args.json or os.path.join(HERE, "results_%s_%s.json" % (args.mode, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(human_summary(result))
    print("saved: %s" % path)


if __name__ == "__main__":
    main()
