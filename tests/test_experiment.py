from pathlib import Path

import numpy as np
import torch

from src.run_experiment import (
    TransformerForecaster,
    chronological_split,
    make_windows,
    moving_block_bootstrap_mae_delta,
    parse_silso_bytes,
    seasonal_naive_from_context,
)


def test_repository_is_research_bundle():
    root = Path(__file__).resolve().parents[1]
    for p in [
        "README.md", "RESEARCH_BUNDLE.md", "DATA.md", "REPRODUCIBILITY.md", "ETHICS.md",
        "src/run_experiment.py", "paper/paper.md", ".github/workflows/ci.yml", ".github/workflows/empirical.yml",
    ]:
        assert (root / p).exists(), p


def test_silso_parser_preserves_definitive_flag_and_removes_sentinel():
    raw = b"1749;01;1749.042;96.7;1.0;5;1\n1749;02;1749.123;-1.0;1.0;5;1\n1749;03;1749.204;40.0;1.0;5;0\n"
    frame = parse_silso_bytes(raw)
    assert len(frame) == 2
    assert frame.iloc[0]["sunspots"] == 96.7
    assert frame["flag"].tolist() == [1, 0]


def test_model_shape():
    model = TransformerForecaster(window=12)
    assert model(torch.zeros(3, 12)).shape == (3,)


def test_window_target_horizon_alignment():
    values = np.arange(20, dtype="float32")
    X, y, idx = make_windows(values, window=5, horizon=3)
    assert X[0].tolist() == [0, 1, 2, 3, 4]
    assert y[0] == 7
    assert idx[0] == 7


def test_chronological_scaler_stops_at_last_training_target():
    values = np.arange(500, dtype="float32")
    split = chronological_split(values, window=24, horizon=6)
    end = split["last_training_target_index"] + 1
    assert np.isclose(split["mean"], values[:end].mean())
    assert 0 < split["val_start"] < split["test_start"]


def test_seasonal_naive_indexing():
    X = np.arange(2 * 24, dtype="float32").reshape(2, 24)
    one = seasonal_naive_from_context(X, mean=0.0, std=1.0, horizon=1)
    twelve = seasonal_naive_from_context(X, mean=0.0, std=1.0, horizon=12)
    assert one[0] == X[0, 12]
    assert twelve[0] == X[0, -1]


def test_moving_block_bootstrap_is_deterministic_and_directional():
    observed = np.arange(60, dtype=float)
    good = observed + 1
    bad = observed + 5
    out1 = moving_block_bootstrap_mae_delta(observed, good, bad, n_boot=50, seed=1)
    out2 = moving_block_bootstrap_mae_delta(observed, good, bad, n_boot=50, seed=1)
    assert out1 == out2
    assert out1["mean_mae_delta_a_minus_b"] < 0
