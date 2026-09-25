# Do Compact Transformers Beat Strong Baselines on Monthly Sunspot Forecasting?

## Abstract

Sequence models are often evaluated against weak baselines at a single horizon. This study tests a compact Transformer on the official WDC-SILSO Version 2.0 monthly mean total sunspot series against persistence, seasonal-naive, ridge and histogram-gradient-boosting forecasts at 1-, 6-, and 12-month horizons. Chronological validation and testing, repeated Transformer seeds, moving-block bootstrap error-difference intervals, high-activity analysis, era robustness and context-length sensitivity are used to determine whether any apparent Transformer advantage is stable.

## Data

The runner downloads the official SILSO CSV, records its SHA-256, and uses only observations marked definitive. SILSO data are CC BY-NC 4.0 and are credited using DOI 10.24414/qnza-ac80.

## Forecast design

A 132-month context is the primary input. Final test windows are the latest 20% of the definitive supervised sequence. A validation tail within the development era selects Transformer checkpoints. Scaling uses training-era observations only.

## Baselines

Persistence and a 12-month seasonal naive provide transparent time-series references. Ridge autoregression provides a high-dimensional linear baseline. Histogram gradient boosting provides a nonlinear non-neural baseline.

## Transformer robustness

The primary architecture is trained at seeds 13, 42 and 73. Horizon-1 context sensitivity is evaluated at 60, 132 and 264 months.

## Evaluation

RMSE and MAE are reported at all horizons. Transformer-vs-baseline MAE differences use a 12-month moving-block bootstrap. Errors are also separated into high-activity versus other periods and early versus late halves of the test era.

## Results

Generated numerical evidence is written to `paper/results.md` and `results/metrics.json`; this methods file deliberately contains no hand-entered winner claim.

## Limitations

The study does not model solar physics, probabilistic predictive intervals or operational space-weather costs. Results are conditional on this data revision, test era, horizon, context and model set.
