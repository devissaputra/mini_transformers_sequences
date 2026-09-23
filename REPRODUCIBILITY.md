# Reproducing the experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The yearly sunspot series is converted into 24-year windows and split chronologically: 228 training windows followed by 57 test windows. Normalization is fitted only on observations available by the end of the training period.

The full experiment compares persistence, Ridge autoregression, and a one-layer Transformer trained for 24 epochs. PyTorch uses seed 42, one CPU thread, and a seeded DataLoader generator.

Outputs are written to `results/metrics.json` and `results/figures/`.

CI runs a one-epoch smoke test plus preprocessing and model-shape tests. Exact Transformer values may move slightly across PyTorch builds.
