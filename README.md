# Transformer Forecasting for Sunspot Activity

[![CI](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/ci.yml)

![Project overview](assets/01_cover.svg)

A small Transformer time-series experiment evaluated against baselines strong enough to challenge it.

> **Does self-attention improve next-year sunspot forecasting over persistence and linear autoregression?**

## Data and evaluation

The project uses the historical yearly sunspot series distributed with statsmodels.

- 24 previous years → next-year target
- 285 windows
- first 228 windows for training
- final 57 windows for testing
- chronological split
- normalization fitted only on the training period

## Models

![Forecasting pipeline](assets/02_data_pipeline.svg)

1. **Persistence:** next year = most recent year.
2. **Ridge autoregression:** linear prediction from the 24-year lag window.
3. **Transformer encoder:** learned scalar projection + positional embedding + one encoder layer.

Transformer configuration:

- model dimension: 24
- attention heads: 4
- feed-forward width: 48
- encoder layers: 1
- dropout: 0.1
- Adam learning rate: 0.002
- 24 epochs

## Recorded results

![Transformer architecture](assets/03_data_or_model.svg)

| Model | RMSE ↓ | MAE ↓ |
|---|---:|---:|
| Persistence | 33.0187 | 25.1982 |
| **Ridge** | **19.2463** | **14.1705** |
| Transformer | 31.8124 | 23.5226 |

![Chronological evaluation](assets/04_evaluation_or_results.svg)

The Transformer improves modestly over persistence, but the much simpler Ridge model is substantially better. With only 228 training windows, the attention model does not earn its extra complexity.

That is the main result of the repo.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

## Test

```bash
pip install pytest
pytest
```

CI checks model shapes, train-only preprocessing, baselines, and a one-epoch Transformer smoke run.

## Why the baseline still matters

The Transformer does beat persistence, but it falls well short of Ridge. With only 228 training windows, that is a useful warning against equating architectural sophistication with better forecasting.

The repo therefore treats the Transformer as one hypothesis among several, not as the default winner. The chronological split and train-only normalization keep the comparison honest.

## What I would try next

A deeper follow-up would add autoregressive statistical models, spectral or seasonal features, rolling-origin evaluation, repeated seeds, validation-based tuning, uncertainty intervals, and alternative forecast horizons. I would only increase Transformer capacity after establishing that the additional data and evaluation design can support it.
