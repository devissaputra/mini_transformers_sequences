from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from src.run_experiment import (
    MAX_EPOCHS,
    EXPECTED_STUDY_INPUT_SHA256,
    TransformerForecaster,
    chronological_split,
    context_sensitivity,
    fixed_target_boundaries,
    freeze_study_frame,
    make_windows,
    moving_block_bootstrap_mae_delta,
    parse_silso_bytes,
    seasonal_naive_from_context,
    select_hgb_baseline,
    select_ridge_baseline,
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
    assert 0 < split["validation_target_start"] < split["test_target_start"]


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


def test_context_lengths_share_target_boundaries():
    values = np.arange(1000, dtype="float32")
    boundaries = fixed_target_boundaries(len(values))
    a = chronological_split(values, window=60, horizon=1, validation_target_start=boundaries[0], test_target_start=boundaries[1])
    b = chronological_split(values, window=264, horizon=1, validation_target_start=boundaries[0], test_target_start=boundaries[1])
    assert a["validation_target_start"] == b["validation_target_start"]
    assert a["test_target_start"] == b["test_target_start"]
    assert a["target_indices_test"][0] == b["target_indices_test"][0]


def test_trainable_parameter_count_does_not_depend_on_context_length():
    a = TransformerForecaster(window=60)
    b = TransformerForecaster(window=264)
    count_a = sum(p.numel() for p in a.parameters() if p.requires_grad)
    count_b = sum(p.numel() for p in b.parameters() if p.requires_grad)
    assert count_a == count_b


def test_frozen_study_input_fingerprint_guard():
    months = pd.date_range("1749-01-01", periods=3327, freq="MS")
    frame = pd.DataFrame({
        "date": months,
        "sunspots": np.arange(3327, dtype=float),
        "flag": np.ones(3327, dtype=int),
    })
    # Synthetic content must not accidentally satisfy the real frozen fingerprint.
    with pytest.raises(ValueError, match="study-input SHA-256"):
        freeze_study_frame(frame)


def test_baseline_selection_uses_declared_validation_grids():
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(80, 8))
    y_train = X_train[:, 0] * 0.5 + rng.normal(scale=0.1, size=80)
    X_val = rng.normal(size=(20, 8))
    y_val = X_val[:, 0] * 0.5 + rng.normal(scale=0.1, size=20)

    _, ridge_meta = select_ridge_baseline(X_train, y_train, X_val, y_val)
    _, hgb_meta = select_hgb_baseline(X_train, y_train, X_val, y_val)

    assert ridge_meta["selected_alpha"] in ridge_meta["candidate_alphas"]
    assert "validation_mse_scaled" in ridge_meta
    assert hgb_meta["selected_learning_rate"] in {0.03, 0.05, 0.10}
    assert hgb_meta["selected_max_leaf_nodes"] in {15, 31}
    assert hgb_meta["early_stopping"] is False
    assert "validation_mse_scaled" in hgb_meta


def test_context_sensitivity_uses_primary_training_budget_by_default():
    assert context_sensitivity.__defaults__[-1] == MAX_EPOCHS


def test_uncertainty_contract_includes_all_primary_baselines():
    declared = {"persistence", "seasonal_naive", "ridge", "hist_gradient_boosting"}
    source = Path(__file__).resolve().parents[1] / "src" / "run_experiment.py"
    text = source.read_text(encoding="utf-8")
    for name in declared:
        assert f'"{name}"' in text
