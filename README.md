# Transformer Forecasting on WDC-SILSO

[![CI](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/ci.yml)
[![Empirical Study](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/empirical.yml/badge.svg)](https://github.com/devissaputra/mini_transformers_sequences/actions/workflows/empirical.yml)

**Research Bundle · AI Engineering · multi-horizon time-series forecasting**

This repository asks a deliberately falsifiable question: **does a compact Transformer earn its complexity over strong simple and tabular baselines on a long real scientific time series?** The source is the official WDC-SILSO Version 2.0 monthly mean total sunspot number. The Transformer is a hypothesis, not the presumed winner.

## Research questions

1. Does a compact Transformer improve 1-, 6-, or 12-month forecasting over persistence, seasonal-naive, ridge autoregression and histogram gradient boosting?
2. Are Transformer results stable across random seeds?
3. Does the conclusion change between early and late parts of the same final test era?
4. Does performance deteriorate disproportionately during high solar activity?
5. Does the result change when context length changes while trainable model size and target dates remain controlled?

## Frozen scientific input

The runner downloads the official `SN_m_tot_V2.0.csv` from WDC-SILSO. SILSO marks CSV rows with `1` for definitive values and `0` for provisional values.

This Research Bundle is frozen to definitive observations from **1749-01 through 2026-03**:

- study rows: **3,327 months**;
- study-input SHA-256: `02adc08ef41aca6e5a02a21417d38bd3ed1dc14ab4bb3de5df5c93488c017a9b`;
- study fingerprint: deterministic `YYYY-MM;sunspot-value` sequence;
- later rows are excluded even after they become definitive.

The full downloaded file hash is still recorded as provenance, but it is not the research identity because SILSO is a living data service. If any month/value pair inside the frozen study era changes, the runner fails.

WDC-SILSO identifies the series as Version 2.0, DOI `10.24414/qnza-ac80`, under CC BY-NC 4.0.

## Controlled chronological design

The comparison uses one common pair of **target-date boundaries** for every horizon and context-length analysis.

- primary context: 132 months;
- horizons: 1, 6 and 12 months;
- final 20% of the frozen raw monthly timeline: test targets;
- final 15% of the pre-test era: validation targets;
- the same validation/test target months are reused across all horizons;
- the same target months are reused for 60-, 132- and 264-month context sensitivity;
- normalization is fitted only through the last training target;
- evaluation is **rolling-origin direct forecasting with observed history**: later test targets may use earlier realized test-era observations in their context, but never the target value or any future observation.

This avoids a subtle confound where changing context length also changes the evaluation era. It also makes clear that the final 20% is not forecast recursively from one fixed historical origin.

## Baselines and Transformer

Baselines:

- persistence;
- 12-month seasonal naive;
- ridge autoregression;
- histogram gradient boosting.

Ridge and histogram gradient boosting are selected from small declared grids using **the same chronological validation era and validation MSE principle** used for Transformer checkpoint selection. HGB's internal random early stopping is disabled, so baseline selection does not introduce a hidden random validation split.

The Transformer uses:

- input projection dimension 32;
- four attention heads;
- feed-forward width 64;
- one encoder layer;
- fixed sinusoidal positional encoding;
- early stopping on the chronological validation era;
- seeds 13, 42 and 73.

Fixed sinusoidal positions are important here: changing context length no longer changes the number of trainable positional parameters.

## Evaluation and robustness

- MAE and RMSE for 1-, 6-, and 12-month horizons;
- repeated Transformer seeds;
- 12-month moving-block bootstrap summaries for Transformer-vs-baseline MAE differences;
- high-activity versus other-period errors using a training-era threshold;
- early-versus-late test-era robustness;
- context sensitivity at 60, 132 and 264 months on identical horizon-1 target dates and the same 60-epoch maximum training budget;
- 12-month moving-block Transformer-vs-baseline MAE intervals for **all four** primary baselines: persistence, seasonal naive, ridge and HGB.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/run_experiment.py
```

Generated evidence includes `results/metrics.json`, `results/summary.md`, three horizon figures, `paper/results.md`, and `paper/results.tex`.

## Interpretation boundary

A win on one horizon does not establish general Transformer superiority. Sunspot dynamics are periodic, nonstationary and physically structured. Results remain conditional on the frozen study input, forecast horizon, evaluation era, baseline set and optimization protocol. A simpler model winning is a valid and useful outcome.

## Professor review path

`README.md` → `DATA.md` → `src/run_experiment.py` → `results/summary.md` → `results/metrics.json` → `RESEARCH_BUNDLE.md` → `REPRODUCIBILITY.md` → `ETHICS.md` → `paper/paper.md`.


## Licensing boundary

Repository source code is released under the MIT License. The WDC-SILSO source data are **not** relicensed by this repository; SILSO identifies them under CC BY-NC 4.0 with attribution requirements. Generated metrics and figures are provided as research evidence, and users should preserve SILSO attribution and assess the source-data license when reusing data-derived artifacts. See `DATA_LICENSE.md`.
