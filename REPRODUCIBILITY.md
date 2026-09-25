# Reproducibility Protocol

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/run_experiment.py
```

## Frozen data identity

- source: WDC-SILSO monthly total Sunspot Number Version 2.0;
- study cutoff: `2026-03`;
- study rows: `3327`;
- model-input SHA-256: `02adc08ef41aca6e5a02a21417d38bd3ed1dc14ab4bb3de5df5c93488c017a9b`;
- definitive rows only inside the frozen era;
- source snapshot SHA-256 recorded separately on every run.

This design allows SILSO to append later observations without silently changing the evidence in this repository.

## Frozen forecasting defaults

- context length: 132 months;
- horizons: 1, 6, 12 months;
- target test boundary: final 20% of the frozen raw timeline;
- target validation boundary: final 15% of the pre-test era;
- identical target-date boundaries across all primary horizons and context sensitivity;
- ridge alpha: 1.0;
- histogram gradient boosting: 300 iterations, learning rate 0.05;
- Transformer projection dimension 32;
- four attention heads;
- feed-forward width 64;
- one encoder layer;
- fixed sinusoidal positional encoding;
- Ridge alpha candidates: 0.1, 1, 10, 100, selected by chronological validation MSE;
- HGB candidate grid: learning rate {0.03, 0.05, 0.10} with max-leaf-nodes {15, 31} in the declared combinations, selected by chronological validation MSE;
- HGB internal early stopping disabled;
- Transformer seeds: 13, 42, 73;
- max epochs: 60;
- early-stopping patience: 8;
- deterministic PyTorch algorithms enabled;
- moving-block bootstrap length: 12 months;
- context sensitivity: 60, 132, 264 months at horizon 1.

## Outputs

A full run writes:

- `results/metrics.json`;
- `results/summary.md`;
- horizon-specific forecast figures;
- `paper/results.md`;
- `paper/results.tex`.

## CI and empirical workflow

Offline CI checks parsing, horizon alignment, scaler boundaries, common target boundaries, trainable-parameter invariance across contexts, seasonal-naive indexing, model shape and bootstrap behavior.

The networked empirical workflow downloads SILSO and regenerates the complete study. Publication is rebase-safe: if newer research code or dependencies reach `main` while a long run is executing, the stale run does not publish results. Documentation-only changes can be incorporated without causing result-file merge conflicts.

## Statistical interpretation

Monthly forecast errors are serially dependent. Twelve-month moving blocks provide a descriptive uncertainty summary for paired MAE differences, not a universal significance test. Repeated model seeds reuse the same historical data and are not independent replications.

## Manuscript build

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The manuscript imports generated `paper/results.tex`; numerical result tables should not be manually transcribed.
