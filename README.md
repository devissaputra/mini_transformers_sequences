# Transformer Forecasting on WDC-SILSO

[![CI](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/ci.yml)
[![Empirical Study](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/empirical.yml/badge.svg)](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/empirical.yml)

**Research Bundle · AI Engineering · multi-horizon sequence forecasting**

This repository asks a deliberately falsifiable question: **does a compact Transformer earn its complexity over strong simple and tabular baselines on a long real scientific time series?** The source is the official WDC-SILSO Version 2.0 monthly mean total sunspot number. The Transformer is a hypothesis, not the presumed winner.

## Research questions

1. Does a compact Transformer improve 1-, 6-, or 12-month forecasting over persistence, seasonal-naive, ridge autoregression, and histogram gradient boosting?
2. Are Transformer results stable across random seeds?
3. Does the conclusion change between early and late parts of the final test era?
4. Does performance deteriorate disproportionately during high solar activity?
5. Does the result depend strongly on the context length?

## Data

The official SILSO monthly mean total sunspot series begins in 1749. The runner downloads `SN_m_tot_V2.0.csv`, records the SHA-256 of the exact bytes used, removes unavailable sentinel values, and uses only rows marked **definitive** for the primary empirical study. Provisional rows are counted and excluded.

WDC-SILSO licenses these data under **CC BY-NC 4.0** and requests explicit attribution to WDC-SILSO, Royal Observatory of Belgium, with DOI `10.24414/qnza-ac80`.

## Frozen design

- context: 132 months;
- horizons: 1, 6 and 12 months;
- final 20% of supervised windows: chronological test set;
- previous 15% of the development era: validation/early stopping;
- normalization fitted only through the final training target;
- baselines: persistence, 12-month seasonal naive, ridge, histogram gradient boosting;
- Transformer seeds: 13, 42, 73;
- test metrics: RMSE and MAE;
- moving-block bootstrap intervals for Transformer-vs-baseline MAE differences;
- high-activity error analysis using the training-era 75th percentile;
- early-vs-late test-era robustness;
- context sensitivity at 60, 132 and 264 months for horizon 1.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/run_experiment.py
```

Generated evidence includes `results/metrics.json`, `results/summary.md`, three horizon forecast figures, and `paper/results.md`.

## Interpretation boundary

A win on one horizon does not establish general Transformer superiority. Sunspot dynamics are periodic, nonstationary and scientifically structured. Conclusions remain conditional on the forecast horizon, test era, context length, baselines, optimization and data revision state. A simpler model winning is a valid and useful outcome.

## Professor review path

`README.md` → `DATA.md` → `src/run_experiment.py` → `results/summary.md` → `results/metrics.json` → `RESEARCH_BUNDLE.md` → `REPRODUCIBILITY.md` → `paper/paper.md`.
