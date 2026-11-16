# Real-device results

JSON files written by `deploy/realtime_test.py` on real sound cards.

| File | Mode | Setup |
|---|---|---|
| `results_file_20261113_153148.json` | A: file playback (wav -> model -> output), vocals output | Apple M2 Pro, macOS 14.6.1, Core Audio, EarPods, block 512, ONNX Runtime 1 thread |
| `results_latency_20261116_143941.json` | B: acoustic loopback click test (8 clicks per path) | Apple M2 Pro, macOS 14.6.1, Core Audio, MacBook Pro Speakers -> MacBook Pro Microphone, block 512 |

The numbers from these runs are quoted in the main README and in `results/summary.csv` (section `device`).
Mode A was stopped manually after 80.6 s (6,907 callbacks in the statistics after 10 warm-up blocks); the timing
statistics cover that period.
