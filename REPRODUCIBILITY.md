# Reproducibility Protocol

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python src/run_experiment.py
```

Frozen defaults:
- SILSO monthly Version 2.0 file
- 132-month context window
- 80% development / 20% final test windows
- within development: 85% fitting / 15% validation
- seed 42
- ridge alpha 1.0
- Transformer dimension 32
- attention heads 4
- feed-forward width 64
- one encoder layer
- max 60 epochs
- patience 8

The real-data runner requires internet access. Unit tests do not.

The prior statsmodels yearly-sunspot metrics are retired and are not research-bundle results.
