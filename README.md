# Transformer Forecasting Research Bundle

[![CI](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/ci.yml)

**Research Bundle · AI Engineering · sequence forecasting with strong baselines**

This repository asks whether a compact Transformer earns its additional complexity on a long, real scientific time series. The default empirical source is the **WDC-SILSO Version 2.0 monthly mean total sunspot number**, fetched from the official SILSO data service.

## Research question

> Does a compact self-attention forecaster improve monthly sunspot prediction over persistence and ridge autoregression under a strictly chronological holdout?

The bundle is designed so a negative result is useful. The Transformer is a hypothesis, not the assumed winner.

## Real dataset

WDC-SILSO, Royal Observatory of Belgium: monthly mean total sunspot number, Version 2.0.

- monthly observations beginning in 1749;
- official source: SILSO data files;
- non-commercial attribution terms apply to the SILSO data;
- recommended data citation/credit is documented in [DATA.md](DATA.md).

The runner downloads the official semicolon-separated file from SILSO and records a SHA-256 checksum of the exact bytes used.

## Frozen protocol

1. Download the monthly Version 2.0 series from SILSO.
2. Remove sentinel negative activity values.
3. Construct 132-month input windows, approximately one solar cycle.
4. Keep the final 20% of windows as an untouched chronological test set.
5. Use the earlier windows for model development:
   - first 85% of the pre-test windows for fitting;
   - last 15% for validation/early stopping.
6. Fit normalization only from the training-era values.
7. Compare:
   - persistence;
   - ridge autoregression;
   - one-layer Transformer encoder.
8. Report RMSE and MAE on the final chronological test segment.
9. Save dataset checksum, observation span, split sizes and software versions.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

## Research Bundle contents

- official external scientific data;
- checksum-based data provenance;
- chronological train/validation/test design;
- strong simple baselines;
- validation-based Transformer checkpoint selection;
- reproducible seeds;
- tests and CI;
- paper-ready protocol;
- explicit limitations and non-claims.

See [RESEARCH_BUNDLE.md](RESEARCH_BUNDLE.md).

## Interpretation boundary

A lower test error on this one series would not establish general Transformer superiority. Sunspot dynamics are structured, periodic and nonstationary; forecasting conclusions depend on horizon, window length, evaluation era and baseline set.

## Citation

When using SILSO data, follow the attribution instructions in [DATA.md](DATA.md). Repository software metadata are in [CITATION.cff](CITATION.cff).
