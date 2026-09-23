# Transformer Forecasting for Sunspot Activity

![Project overview](assets/01_cover.svg)

I built this project to test whether a small Transformer can handle a real time-series forecasting problem without using a recurrent network.

The task is to use the previous 24 yearly sunspot observations to predict the next year.

## Data

I use the historical sunspot dataset distributed with statsmodels.

The sequence construction produces:

- 285 total windows;
- 228 training windows;
- 57 held-out windows;
- 24 years of history per input;
- one next-year target.

The split is chronological. Normalization is fitted on the training period only and then applied to the later hold-out period.

## How the model works

![Forecasting pipeline](assets/02_data_pipeline.svg)

Each scalar sunspot value is projected into a 24-dimensional representation. I add a learned positional embedding so the model can distinguish one year in the input window from another.

The encoder uses:

- model dimension 24;
- 4 attention heads;
- feed-forward width 48;
- 1 encoder layer;
- dropout 0.1.

The final token representation is passed through a linear layer to produce the next-year forecast.

## Transformer structure

![Transformer architecture](assets/03_data_or_model.svg)

Unlike the LSTM in the previous project, self-attention allows each position in the 24-year input window to compare directly with every other position.

This does not automatically make the Transformer better. With only a few hundred windows, model size and validation design matter a lot.

## Results

![Chronological evaluation](assets/04_evaluation_or_results.svg)

After correcting preprocessing so normalization uses the training period only, the recorded run produced:

| Metric | Result |
|---|---:|
| RMSE | 33.2499 |
| MAE | 23.8058 |
| Training windows | 228 |
| Test windows | 57 |

These errors are fairly large, which is useful information rather than something to hide. The series is small and strongly variable, so a Transformer is not automatically the right tool.

A serious follow-up should compare this result against simple baselines, autoregressive models, and the LSTM project using the same evaluation window.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

On Windows, use `.venv\Scripts\activate`.

## Repository notes

- [DATA.md](DATA.md) explains the sunspot series.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) explains the temporal split and rerun steps.
- [paper/paper.md](paper/paper.md) contains the longer technical write-up.
