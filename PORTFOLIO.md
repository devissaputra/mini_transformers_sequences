# Transformer Forecasting for Sunspot Activity

**Focus:** testing whether self-attention earns its complexity on a small time series.

A one-layer Transformer forecasts next-year sunspot activity from the previous 24 years and is evaluated against persistence and Ridge autoregression on a chronological hold-out set. Normalization is fitted only on the training period.

The Transformer improves modestly over persistence: RMSE 31.8124 versus 33.0187. Ridge is substantially better at 19.2463. With only 228 training windows, the linear autoregressive model is the clear choice in this configuration.

The repository includes deterministic PyTorch training, strong baselines, behavioural tests, CI, reproducibility notes, and generated diagnostics.
