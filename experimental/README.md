# Comparison experiments

Code for the models that F12 was compared against. None of it is needed to train, evaluate or deploy F12; it is
kept so that the comparison in the main README can be reproduced. All models here are causal and have a
frame-by-frame `step()` path that matches the whole-signal `forward()` (checked in `tests/`).

| File | Model | Status |
|---|---|---|
| `music_sparse_v1.py`, `music_sparse_v2.py` | MusicSparse: STFT band-split network inspired by SCNet's sparse band compression (three frequency regions compressed 1x / 4x / 16x, cross-band convolution, local and global causal GRUs, complex mask). Not an SCNet reproduction. | v2 (FFT 1024, 23.2 ms) reached 4.02 dB valid SDR |
| `fast_music.py` | MusicSparse v2 variants: `fft512` = **model C** (FFT 512, hop 128, 11.6 ms); `stride8` (coarser high band, 23.2 ms) | C tested once on test: 2.51 dB |
| `asym_music.py` | **Model S**: MusicSparse v2 with an asymmetric analysis / synthesis window pair (1024-sample analysis, 512-sample synthesis support, hop 128), giving the frequency resolution of a 1024 FFT at the latency of a 512 window (11.6 ms) | tested once on test: 2.58 dB |
| `fast_music_ort.py` | ONNX export of C as a fixed 512-sample block step with explicit state (real-valued DFT) | used for C's test and timing |
| `rtstt.py`, `rtstt_paper.py`, `rtstt_chunked.py` | Adaptation of RT-STT (real-time single-path TFC-TDF UNet, arXiv 2511.13146), four-source output, 23.2 ms. Building blocks adapted from DTTNet-Pytorch (Apache-2.0). Not an exact reproduction; deviations are listed in `configs/rtstt_train.json`. | 4.04 dB valid SDR (valid only, no test) |
| `train_music_sparse.py` | training for C / S / v2 / stride8 | |
| `train_rtstt.py`, `configs/rtstt_train.json` | training for the RT-STT adaptation | |

Valid SDR = museval v4 vocals SDR on the 14 whole valid tracks with block-512 ONNX streaming (same protocol as F12,
whose valid SDR is 4.75 dB). Test numbers use the main README protocol.

The asymmetric-window result (S vs C: +0.85 dB on valid, +0.07 dB on test) comes from a single seed per model and
should be treated as unconfirmed.

```bash
python -m experimental.train_music_sparse --variant fft512 --out runs/C
python -m experimental.train_music_sparse --variant asym --out runs/S
python -m experimental.train_rtstt --out runs/rtstt
python scripts/evaluate.py --ckpt runs/S/best.pt --out eval/S_valid        # architecture inferred from the checkpoint
```
