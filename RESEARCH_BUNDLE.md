# Research Bundle Evidence Contract

## Identity

**Area:** AI Engineering  
**Study:** compact Transformer versus strong forecasting baselines across horizons  
**Dataset:** WDC-SILSO monthly mean total sunspot number, Version 2.0

## Required evidence

A valid full result records:

1. official SILSO URL, DOI, license and SHA-256;
2. raw nonmissing row count, definitive row count and excluded provisional row count;
3. first and last definitive month;
4. context length and forecast horizons;
5. chronological train/validation/test boundaries and scaler fit boundary;
6. persistence, seasonal-naive, ridge and histogram-gradient-boosting baselines;
7. Transformer architecture, seeds and best validation epochs;
8. RMSE and MAE at 1-, 6-, and 12-month horizons;
9. moving-block bootstrap intervals of Transformer-vs-baseline absolute-error differences;
10. high-activity versus other-period error analysis;
11. early-versus-late test-era robustness;
12. context-length sensitivity;
13. software environment and generated figures.

## Statistical boundary

Forecast errors are serially dependent. The repository therefore avoids an iid bootstrap and uses 12-month moving blocks for descriptive paired error-difference intervals. These intervals do not establish universal statistical superiority.

## Non-claims

The bundle does not claim general Transformer superiority, causal modeling of solar physics, operational space-weather readiness, or transfer to unrelated sequence domains.
