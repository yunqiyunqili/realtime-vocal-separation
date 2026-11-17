# Real-time test on a laptop sound card

`realtime_test.py` runs the F12 causal Conv-TasNet (44.1 kHz stereo) on a real audio device and measures whether it
keeps up in real time. It needs only four libraries: numpy, soundfile, sounddevice and onnxruntime (no PyTorch).

| File | Where it comes from |
|---|---|
| `realtime_test.py` | this directory |
| `requirements.txt` | this directory |
| `f12_stream_block512.onnx`, `model_meta.json` | not publicly released yet (available from the author on request); with a checkpoint: `python scripts/export_onnx.py --ckpt f12_model.pt --out-dir deploy/` |
| a 44.1 kHz music file | your own (no audio is shipped: MUSDB18-HQ may not be redistributed) |

The ONNX model takes one block of 512 samples per call; all model state is explicit input/output
(`model_meta.json` lists names, shapes and the 384-sample output delay).

## 1. Install

Python 3.9 or later.

```bash
cd deploy
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
python realtime_test.py --list-devices      # sound cards and host APIs
```

- **Linux:** PortAudio and libsndfile must be installed, e.g. `sudo apt install libportaudio2 libsndfile1`.
- **Windows / macOS:** the pip wheels bundle both libraries.

## 2. Mode A: file -> model -> speakers (no microphone needed)

```bash
python realtime_test.py --wav song.wav                    # outputs vocals for 120 s (the file loops)
python realtime_test.py --wav song.wav --output mixture --duration 60
python realtime_test.py --wav song.wav --device "Headphones"
```

- **How it works:** opens a sounddevice output stream with blocksize 512. Every callback reads 512 samples, runs one
  ONNX Runtime inference (1 intra-op and 1 inter-op thread) and writes the result to the device.
- **Switching output:** type `v` / `m` / `a` + Enter to switch between vocals (separated), mixture (original) and
  accompaniment (mixture - vocals); `q` + Enter quits.
- **Alignment:** the model output lags its input by 384 samples; mixture and accompaniment go through the same
  delay, so switching is seamless.
- **Inference always runs:** the model runs on every block whatever you listen to, so timing does not depend on
  the output choice.

## 3. Mode B: end-to-end latency estimate (optional)

```bash
python realtime_test.py --mode latency
python realtime_test.py --mode latency --device "USB Audio" --input-device "USB Audio"
```

- **Procedure:** opens a full-duplex stream, plays two kinds of clicks alternately (8 each, 0.5 s apart by default)
  and records them with the microphone. The round trip of each click is estimated by cross-correlation and the
  median is reported. A click is a 20 ms exponential sweep (300 Hz-8 kHz), which gives a sharper, more
  noise-robust correlation peak than a single impulse.
  - **Passthrough:** the click is written straight to the output.
  - **Through the model:** the click is the model input and the model's accompaniment output (aligned mixture -
    vocals estimate) is played, so it carries the full model delay. The vocals output is not used because the
    model may remove a click that does not sound like a voice.
- **Result:** the difference is the latency added by the model. Theory: the 384-sample output delay = 8.7 ms.
  - the 512-sample block buffering exists on both paths and cancels in the difference;
  - inference time adds no latency as long as each callback finishes within its budget; if it does not, the
    symptom is an xrun, not a longer delay;
  - the 11.6 ms algorithmic latency is the upper bound set by the 512-sample model window.
- **Also printed:** the input and output latency that sounddevice reports (`stream.latency`), to compare with the
  measured passthrough round trip.
- **Accuracy:** a laptop speaker -> built-in microphone loop is noisy (echo cancellation, noise suppression or
  automatic gain can interfere, and clicks may be missed). A cable loopback from line/headphone out to line in is
  better. A click counts only if its correlation peak is >= 8x the median (`--min-confidence`).
- **macOS:** allow the terminal to use the microphone on first run.

## 4. Reading the results

Each run prints a summary and saves `results_<mode>_<time>.json` (or `--json PATH`) with the CPU model, OS, device
name and host API, sample rate, blocksize and:

| Metric | Meaning | Expected |
|---|---|---|
| `inference_ms` p50 / p99 / max | ONNX inference time per block (first 10 blocks excluded as warm-up) | p99 well below 11.61 ms |
| `inference_over_budget` | blocks whose inference exceeded 11.61 ms | 0 or very few |
| `callback_ms` | whole callback time (including copies) | slightly above inference |
| `rtf_median` | median inference time / 11.61 ms | < 1 |
| `xruns_output_underflow` | output underflows reported by the driver (audible dropouts) | **0** |
| `status_flags` | other PortAudio status flags (input_overflow etc.) | 0 |
| mode B `passthrough` / `through_model` | median / min / max round trip and number of reliable clicks | difference ~8.7 ms |

- **Pass criterion:** real-time capability is judged mainly by zero xruns. A block over budget does not
  necessarily cause an xrun because drivers usually buffer more.
- **Background load:** close CPU-heavy programs, plug in the laptop and use a high-performance power mode.

## 5. Troubleshooting

- **Device not found / wrong device:** run `--list-devices`, then `--device <index>` or `--device "<part of name>"`;
  `--hostapi WASAPI` (or `ASIO`, `Core Audio`, `ALSA`) restricts the host API. Mode B's input is set separately with
  `--input-device`.
- **Sample-rate mismatch ("Invalid sample rate" etc.):** the model supports 44100 Hz only.
  - Convert other files first, e.g. `ffmpeg -i in.wav -ar 44100 out.wav`.
  - If the device does not run at 44100 Hz, change its default format in the OS (Windows: Sound settings -> device
    properties -> Advanced -> Default format; macOS: Audio MIDI Setup).
- **WASAPI on Windows:** MME and DirectSound have high, jittery latency; use `--hostapi WASAPI`.
  - WASAPI shared mode requires the Windows mixer rate, so set the default format to 44100 Hz, or add
    `--wasapi-exclusive` to bypass the mixer (also lower latency).
  - Devices with an ASIO driver can use `--hostapi ASIO` with an ASIO-enabled PortAudio build.
- **Dropouts / many xruns:** add `--latency high` or `--latency 0.05` to let the driver buffer more (round-trip
  latency grows accordingly); close other programs; check power-saving mode.
- **Mode B finds no clicks:** raise the system volume or `--click-amp`, move closer to the microphone, disable
  system noise suppression / echo cancellation, or use a cable loopback.
- **Smoke test without a sound card:** `python realtime_test.py --simulate 1000` calls the audio callback 1000
  times directly (uses 30 s of noise if no `--wav` is given); `python realtime_test.py --mode latency --simulate 1`
  runs the latency estimator on a simulated 3000-sample loopback (expected: passthrough 68.0 ms, through model
  76.7 ms).

## 6. Notes

- The server timing in the main README (Xeon Gold 6150, exclusive node) and laptop results differ with CPU and
  power policy.
- Tested on real hardware so far: Apple M2 Pro with Core Audio only. Windows and Linux devices are untested.
