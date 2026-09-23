# Transformer Forecasting for Sunspot Activity

## Abstract

This experiment evaluates a compact Transformer against persistence and Ridge autoregression for next-year sunspot forecasting. The data are split chronologically and normalization is fitted only on the training period.

## Method

Twenty-four prior annual observations predict the next year. The experiment contains 285 windows, with 228 used for training and 57 held out for evaluation. The Transformer uses a 24-dimensional representation, four attention heads, one encoder layer, and a 48-unit feed-forward block.

## Results

| Model | RMSE | MAE |
|---|---:|---:|
| Persistence | 33.0187 | 25.1982 |
| Ridge | 19.2463 | 14.1705 |
| Transformer | 31.8124 | 23.5226 |

## Interpretation

The Transformer modestly improves over persistence but is substantially worse than Ridge. With only a few hundred sequence windows, self-attention does not justify its added complexity in this configuration.

## Limitations

The study uses one series, one hold-out period, and one Transformer configuration. Rolling-origin evaluation, stronger statistical baselines, repeated seeds, uncertainty estimates, and validation-based tuning are needed for stronger conclusions.
