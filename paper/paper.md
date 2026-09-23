# Transformer Forecasting for Sunspot Activity

## Question

Can a small Transformer encoder forecast the next yearly sunspot value from the previous 24 years?

## Data

I use the yearly sunspot series from statsmodels.

The series produces 285 sliding windows. I use the first 228 for training and the final 57 for chronological hold-out evaluation.

Normalization statistics are fitted on the training period only.

## Method

Each scalar observation is projected into 24 dimensions and combined with a learned positional embedding.

The encoder uses:

- model dimension 24;
- 4 attention heads;
- feed-forward width 48;
- 1 encoder layer;
- dropout 0.1.

The final token representation is mapped to one forecast value.

Training uses Adam with learning rate 0.002, mean-squared error loss, batch size 32, and 24 epochs.

## Results

After correcting normalization to use the training period only, the recorded run produced:

| Metric | Result |
|---|---:|
| RMSE | 33.2499 |
| MAE | 23.8058 |

## Interpretation

The error is substantial. I consider that an important result, because a more complex model does not automatically make a better forecaster.

The dataset is small, and a Transformer has many ways to spend model capacity. Before changing the architecture, I would compare it with simple baselines and the LSTM project using exactly the same hold-out period.

## Limitations

The experiment uses one historical series and one chronological split. There is no baseline model in the current repository, and the sample size is small for a Transformer.

A stronger version would add rolling-origin evaluation, simple and autoregressive baselines, several random seeds, and an attention analysis to see what years the model focuses on.

## Reproduce

```bash
pip install -r requirements.txt
python src/run_experiment.py
```
