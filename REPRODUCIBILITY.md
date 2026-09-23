# Reproducing the Experiment

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The script sets the PyTorch seed to 42 and uses one CPU thread.

The yearly sunspot series is converted into 24-year windows and split chronologically: 80% for training and 20% for evaluation.

Normalization is fitted only on the training period and then applied unchanged to the hold-out period.

The Transformer trains for 24 epochs with Adam at learning rate 0.002.

Results are saved to `results/metrics.json`. PyTorch versions can produce small numerical differences, so record the package versions if exact comparison matters.
