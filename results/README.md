# Results

- `summary.csv`: every number used in the main README, one row per (model, metric, split/condition).
  `source` names the result file in the original experiment directory the value was read from (those raw logs,
  per-track JSONs and checkpoints are not part of this repository), or the device run for the laptop numbers.
  Timing rows give the median of three repetitions; the individual repetitions are listed in `note`.
- `latency_vs_sdr.png`: produced by `python scripts/make_figure.py` from `summary.csv`.
- `device/`: real sound-card runs of `deploy/realtime_test.py`.
