from pathlib import Path

import numpy as np
import torch

from src.run_experiment import (
    TransformerForecaster,
    prepare_windows,
    run_experiment,
)


def test_model_output_shape():
    model = TransformerForecaster()
    model.eval()
    output = model(torch.zeros(4, 24))
    assert output.shape == (4,)


def test_normalization_uses_training_period_only():
    values = np.arange(60, dtype="float32")
    _, _, split, mean, _ = prepare_windows(values, window=6, train_fraction=0.8)
    expected_end = split + 6
    assert mean == float(values[:expected_end].mean())


def test_one_epoch_smoke_run(tmp_path):
    result = run_experiment(tmp_path, epochs=1, make_plots=False)
    assert set(["persistence", "ridge", "transformer"]).issubset(result)
    assert result["test_windows"] == 57
    for name in ["persistence", "ridge", "transformer"]:
        assert result[name]["rmse"] >= 0
        assert result[name]["mae"] >= 0


def test_repository_structure():
    root = Path(__file__).resolve().parents[1]
    assert (root / ".github/workflows/ci.yml").exists()
    assert (root / "paper/paper.md").exists()
