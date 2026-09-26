# Calculation guide

## Question and evidence

Does a small Transformer justify its complexity?

WDC-SILSO monthly sunspots, January 1749 to March 2026: 3,327 definitive observations.

**Status:** RECORDED EXTERNAL-DATA STUDY | full experiment not rerun in this review.

## Design

Common chronological target dates; validation-selected baselines; 1-, 6- and 12-month direct forecasts; three Transformer seeds.

## Calculation and interpretation

`MAE = mean(|forecast - observed|); delta = Transformer MAE - baseline MAE.`

Negative paired MAE differences favor the Transformer. Twelve-month moving blocks retain some serial dependence. Later test predictions may use already observed test-era history; this is rolling-origin forecasting, not one fixed-origin recursive forecast.

## Evidence table

Selected recorded values (units and context shown). Full precision below is for traceability, not a claim of measurement precision.

| Quantity | Value | Unit / meaning | JSON path |
|---|---:|---|---|
| 1 month Transformer mean | 15.913768871410474 | sunspot MAE ↓ | `horizons.1.metrics.transformer_repeated_seed_summary.mae_mean` |
| 1 month HGB | 16.90527831823438 | sunspot MAE ↓ | `horizons.1.metrics.hist_gradient_boosting.mae` |
| 6 month Transformer mean | 19.69880894688634 | sunspot MAE ↓ | `horizons.6.metrics.transformer_repeated_seed_summary.mae_mean` |
| 6 month HGB | 20.537293415179263 | sunspot MAE ↓ | `horizons.6.metrics.hist_gradient_boosting.mae` |
| 12 month Transformer mean | 24.206205192390268 | sunspot MAE ↓ | `horizons.12.metrics.transformer_repeated_seed_summary.mae_mean` |
| 12 month HGB | 23.97272047251438 | sunspot MAE ↓ | `horizons.12.metrics.hist_gradient_boosting.mae` |

Source: [results/metrics.json](results/metrics.json). Values resolve directly from this file when figures are regenerated.

The Transformer has lower mean error than the simpler baselines at the 1- and 6-month horizons under the recorded protocol. The 12-month result is less stable: its three-seed mean MAE is 24.206, versus 23.973 for histogram gradient boosting, despite seed 42 favoring the Transformer. The seed-42 intervals against that baseline also cross zero at 6 and 12 months. This qualifies any broad claim that the Transformer is consistently superior.

## Verification performed in this review

The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review. Stored empirical results were inspected, not independently reproduced from raw data.

The figure-generation check verifies agreement between the selected source values and SVGs. It does not validate the raw dataset, fitted model, identification assumptions, or external generalization.

```bash
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

## Implementation map

Follow these functions to inspect each transformation. Validation helpers and private functions remain visible in the linked modules.

| Function | Purpose / documented behavior |
|---|---|
| [`parse_silso_bytes`](src/run_experiment.py#L77) | Inspect the explicit implementation and its callers. |
| [`study_input_fingerprint`](src/run_experiment.py#L95) | Inspect the explicit implementation and its callers. |
| [`freeze_study_frame`](src/run_experiment.py#L103) | Inspect the explicit implementation and its callers. |
| [`load_real_series`](src/run_experiment.py#L129) | Inspect the explicit implementation and its callers. |
| [`make_windows`](src/run_experiment.py#L177) | Inspect the explicit implementation and its callers. |
| [`fixed_target_boundaries`](src/run_experiment.py#L188) | Inspect the explicit implementation and its callers. |
| [`chronological_split`](src/run_experiment.py#L196) | Inspect the explicit implementation and its callers. |
| [`error_metrics`](src/run_experiment.py#L240) | Inspect the explicit implementation and its callers. |
| [`select_ridge_baseline`](src/run_experiment.py#L249) | Inspect the explicit implementation and its callers. |
| [`select_hgb_baseline`](src/run_experiment.py#L265) | Inspect the explicit implementation and its callers. |
| [`seasonal_naive_from_context`](src/run_experiment.py#L292) | Inspect the explicit implementation and its callers. |
| [`activity_error_analysis`](src/run_experiment.py#L299) | Inspect the explicit implementation and its callers. |
| [`moving_block_bootstrap_mae_delta`](src/run_experiment.py#L313) | Inspect the explicit implementation and its callers. |
| [`train_transformer`](src/run_experiment.py#L335) | Inspect the explicit implementation and its callers. |
| [`fit_horizon`](src/run_experiment.py#L377) | Inspect the explicit implementation and its callers. |
| [`context_sensitivity`](src/run_experiment.py#L479) | Inspect the explicit implementation and its callers. |
| [`write_figures`](src/run_experiment.py#L507) | Inspect the explicit implementation and its callers. |
| [`build_results_latex`](src/run_experiment.py#L530) | Inspect the explicit implementation and its callers. |
| [`write_summary`](src/run_experiment.py#L642) | Inspect the explicit implementation and its callers. |
| [`run_experiment`](src/run_experiment.py#L735) | Inspect the explicit implementation and its callers. |
| [`main`](src/run_experiment.py#L796) | Inspect the explicit implementation and its callers. |
| [`forward`](src/run_experiment.py#L72) | Inspect the explicit implementation and its callers. |

## What remains before a stronger research claim

Negative paired MAE differences favor the Transformer. Twelve-month moving blocks retain some serial dependence. Later test predictions may use already observed test-era history; this is rolling-origin forecasting, not one fixed-origin recursive forecast. A successful software test is not validation of a scientific construct. New experiments should state their split unit, comparator, outcome, uncertainty procedure and failure criteria before examining final test results.
