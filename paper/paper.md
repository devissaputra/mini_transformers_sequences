# Do Compact Transformers Beat Linear Baselines on Long-Horizon Sunspot History?

## Status
Research-bundle manuscript scaffold. Numerical findings are generated from the current WDC-SILSO runner.

## Question
Does a compact Transformer improve monthly sunspot forecasting over persistence and ridge autoregression under chronological holdout?

## Data
WDC-SILSO monthly mean total sunspot number, Version 2.0. The runner records the exact upstream SHA-256 checksum used for each study run.

## Design
A 132-month context window is used. The final 20% of windows are held out chronologically. A validation tail inside the development era selects the Transformer checkpoint. Normalization is fitted only from the fitting-era values.

## Evaluation
RMSE and MAE are reported for persistence, ridge and Transformer. A result in which the simpler baseline wins is treated as scientifically informative.

## Limitations
This benchmark does not model solar physics, uncertainty intervals, regime shifts or operational space-weather consequences.
