# Research Bundle Evidence Contract

## Identity

**Area:** AI Engineering  
**Study:** compact Transformer versus strong forecasting baselines across horizons  
**Dataset:** WDC-SILSO monthly mean total Sunspot Number Version 2.0  
**Frozen study era:** 1749-01 through 2026-03  
**Study-input SHA-256:** `02adc08ef41aca6e5a02a21417d38bd3ed1dc14ab4bb3de5df5c93488c017a9b`

## Required evidence

A valid full result records:

1. official SILSO source, DOI, license, downloaded-file SHA-256 and frozen study-input SHA-256;
2. frozen study cutoff, study-row count, latest available/definitive months and excluded post-cutoff/provisional rows;
3. monthly continuity of the frozen study series;
4. common validation and test target-date boundaries;
5. context length and forecast horizons;
6. scaler fit boundary restricted to the training era;
7. persistence and seasonal-naive baselines plus Ridge/HGB candidates selected on the chronological validation era by MSE;
8. HGB internal random early stopping disabled and selected baseline hyperparameters recorded;
9. Transformer architecture, fixed positional encoding, seeds and best validation epochs;
10. RMSE and MAE at 1-, 6-, and 12-month horizons;
11. moving-block bootstrap summaries of Transformer-vs-baseline absolute-error differences for persistence, seasonal naive, Ridge and HGB;
12. high-activity versus other-period error analysis;
13. early-versus-late test-era robustness;
14. context-length sensitivity on identical target dates with invariant trainable Transformer size;
15. software environment and generated figures;
16. synchronized Markdown and LaTeX generated results.

## Comparison contract

Context sensitivity must change context history without changing the test target dates, the maximum Transformer training budget or the number of trainable positional parameters. Primary horizons must also share the same validation and test target boundaries.

The primary evaluation is rolling-origin direct forecasting with observed history. Earlier realized test-era observations may enter the context for later test targets; future observations and target values may not.

## Statistical boundary

Forecast errors are serially dependent. Moving-block bootstrap intervals are descriptive paired error-difference summaries. Transformer seeds quantify optimization sensitivity on the same historical series and are not independent replications.

## Non-claims

The bundle does not claim general Transformer superiority, causal modeling of solar physics, operational space-weather readiness, or transfer to unrelated sequence domains.
