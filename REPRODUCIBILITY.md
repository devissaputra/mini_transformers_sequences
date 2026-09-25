# Reproducibility Protocol

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/run_experiment.py
```

## Frozen defaults

- WDC-SILSO monthly Version 2.0 CSV;
- definitive observations only;
- context length: 132 months;
- horizons: 1, 6, 12 months;
- final test fraction: 0.20;
- validation fraction of development era: 0.15;
- ridge alpha: 1.0;
- histogram gradient boosting: 300 iterations, learning rate 0.05;
- Transformer dimension 32, 4 heads, feed-forward width 64, one encoder layer;
- Transformer seeds: 13, 42, 73;
- max epochs: 60;
- early-stopping patience: 8;
- moving-block bootstrap length: 12 months;
- context sensitivity: 60, 132, 264 months at horizon 1.

## Data identity

Every run records the SHA-256 of the exact SILSO CSV response and the final definitive month. This matters because the upstream time series is extended and recent values may be revised.

## Outputs

The full run writes `results/metrics.json`, `results/summary.md`, horizon-specific forecast figures, and `paper/results.md`.

## CI boundary

Offline tests check source parsing, target-horizon alignment, chronology, normalization boundaries, seasonal-naive indexing, Transformer shape and moving-block bootstrap behavior. The separate empirical workflow performs the networked SILSO run and commits generated evidence.

## Result policy

Metrics from the retired yearly-series demonstration or earlier one-horizon protocol are not evidence for this bundle.
