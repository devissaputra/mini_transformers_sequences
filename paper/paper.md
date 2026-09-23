# Transformer for Real Temporal Sequences: Scientific-Style Technical Report

**Status:** reproducible portfolio report, not peer reviewed.  
**Dataset:** Sunspots dataset via statsmodels  
**Difficulty:** ★★★★★

## Abstract
Train a compact Transformer encoder to forecast real yearly sunspot activity from historical windows. The repository emphasizes traceable data processing, reproducible implementation, task-appropriate evaluation, and explicit limitations.

## Method
1. Load sunspots
2. Window sequences
3. Normalize
4. Transformer encoder
5. Temporal forecast

## Evaluation
Primary metric(s): RMSE / MAE. Validation: chronological hold-out.

## Results
```json
{
  "rmse": 33.0650634765625,
  "mae": 23.189212799072266,
  "window_years": 24,
  "n_windows": 285
}
```

## Limitations
small historical series. Results on one public benchmark do not establish universal model quality.

## Reproducibility
Install `requirements.txt` and run `python src/run_experiment.py`.

## Reference
https://www.statsmodels.org/stable/datasets/generated/sunspots.html
