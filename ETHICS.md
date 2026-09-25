# Ethics, Scientific Limits and Responsible Interpretation

This repository uses a public scientific time series and contains no personal data. Its main risks are methodological overclaiming, inappropriate reuse of licensed data and false confidence in a complex model.

## Complexity is not evidence

A Transformer is not treated as superior because it is newer or more complex. The bundle includes persistence, seasonal-naive, linear and nonlinear non-neural baselines and accepts a negative Transformer result as scientifically informative.

## Fair comparison boundary

Every primary horizon and every context-length sensitivity condition uses common target-date boundaries. Context length changes the available history, not the target era, and context-sensitivity runs use the same maximum epoch cap and early-stopping policy as the primary study. Fixed sinusoidal positional encoding also keeps the trainable Transformer parameter count independent of context length. The evaluation is rolling-origin with observed history, not a single fixed-origin recursive forecast.

Context length still changes attention compute and the number of usable training windows, so these sensitivity conditions do not claim equal wall-clock or FLOP budgets.

## Retrospective activity diagnostic

The high-activity error analysis groups test months using their realized sunspot values relative to a threshold estimated from training data. This is a retrospective diagnostic of where forecast errors concentrate; it is not a regime label available to the forecasting models at prediction time.

## Forecast scope

The experiment is not a physical model of the Sun and is not validated for operational space-weather decisions. Forecast errors may differ substantially around solar maxima, across eras and under future regime changes.

## Data revision and license

SILSO is a living dataset. This bundle freezes its study input through March 2026 using a deterministic month/value fingerprint; later rows do not silently alter the evidence. SILSO data are CC BY-NC 4.0, so downstream users must preserve attribution and respect the non-commercial condition.

## Statistical limits

Monthly forecast errors are serially dependent. Moving-block bootstrap summaries reduce the mismatch of an iid bootstrap but remain descriptive rather than universal significance guarantees. Repeated Transformer seeds reuse the same historical sequence and are not independent replications.
