from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import platform
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

PRIMARY_SEED = 42
TRANSFORMER_SEEDS = (13, 42, 73)
WINDOW = 132
HORIZONS = (1, 6, 12)
DATA_URL = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"
DATA_DOI = "10.24414/qnza-ac80"
DATA_LICENSE = "CC BY-NC 4.0"
MAX_EPOCHS = 60
PATIENCE = 8
STUDY_CUTOFF_MONTH = "2026-03"
EXPECTED_STUDY_ROWS = 3327
EXPECTED_STUDY_INPUT_SHA256 = "02adc08ef41aca6e5a02a21417d38bd3ed1dc14ab4bb3de5df5c93488c017a9b"
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
HGB_CANDIDATES = (
    {"learning_rate": 0.03, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_leaf_nodes": 31},
    {"learning_rate": 0.10, "max_leaf_nodes": 15},
)


class TransformerForecaster(nn.Module):
    def __init__(self, window: int = WINDOW):
        super().__init__()
        dim = 32
        self.projection = nn.Linear(1, dim)
        position = torch.arange(window, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-np.log(10000.0) / dim)
        )
        pe = torch.zeros(window, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("position", pe.unsqueeze(0), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=64,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        z = self.projection(x.unsqueeze(-1)) + self.position
        return self.head(self.encoder(z)[:, -1]).squeeze(-1)


def parse_silso_bytes(raw: bytes) -> pd.DataFrame:
    frame = pd.read_csv(
        io.BytesIO(raw),
        sep=";",
        header=None,
        names=["year", "month", "decimal_date", "sunspots", "std", "n_obs", "flag"],
        engine="python",
    )
    for column in ("year", "month", "decimal_date", "sunspots", "std", "n_obs", "flag"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["year", "month", "decimal_date", "sunspots"])
    frame = frame[frame["sunspots"] >= 0].reset_index(drop=True)
    frame["date"] = pd.to_datetime(
        {"year": frame["year"].astype(int), "month": frame["month"].astype(int), "day": 1}
    )
    return frame


def study_input_fingerprint(frame: pd.DataFrame) -> str:
    payload = "".join(
        f"{row.date.strftime('%Y-%m')};{float(row.sunspots):.1f}\n"
        for row in frame.itertuples()
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_study_frame(full: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cutoff = pd.Timestamp(f"{STUDY_CUTOFF_MONTH}-01")
    study = full[(full["flag"] == 1) & (full["date"] <= cutoff)].copy().reset_index(drop=True)
    if len(study) != EXPECTED_STUDY_ROWS:
        raise ValueError(
            f"Unexpected frozen SILSO study length: {len(study)}; expected {EXPECTED_STUDY_ROWS}"
        )
    if study.empty or study["date"].iloc[-1] != cutoff:
        raise ValueError(
            f"Frozen SILSO study must end at {STUDY_CUTOFF_MONTH}; "
            f"got {study['date'].iloc[-1] if not study.empty else 'empty'}"
        )
    expected_dates = pd.date_range(study["date"].iloc[0], cutoff, freq="MS")
    if len(expected_dates) != len(study) or not np.array_equal(
        study["date"].to_numpy(), expected_dates.to_numpy()
    ):
        raise ValueError("Frozen SILSO study series is not a complete monthly sequence")
    fingerprint = study_input_fingerprint(study)
    if fingerprint != EXPECTED_STUDY_INPUT_SHA256:
        raise ValueError(
            f"Unexpected frozen SILSO study-input SHA-256: {fingerprint}; "
            f"expected {EXPECTED_STUDY_INPUT_SHA256}"
        )
    return study, fingerprint


def load_real_series(data_path: str | Path | None = None, cache_dir: str | Path = "data/cache"):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / "SN_m_tot_V2.0.csv"
    if data_path is not None:
        raw = Path(data_path).read_bytes()
        source = f"local:{data_path}"
    elif cached.exists():
        raw = cached.read_bytes()
        source = f"cache:{cached}"
    else:
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            raw = response.read()
        cached.write_bytes(raw)
        source = DATA_URL

    full = parse_silso_bytes(raw)
    if full["flag"].isna().any():
        raise ValueError("SILSO CSV contains missing or non-numeric definitive/provisional flags")
    invalid_flags = sorted(set(full["flag"].astype(int).tolist()) - {0, 1})
    if invalid_flags:
        raise ValueError(f"Unexpected SILSO definitive/provisional flags: {invalid_flags}")
    available_definitive = full[full["flag"] == 1].copy().reset_index(drop=True)
    if len(available_definitive) < 1000:
        raise ValueError("unexpectedly few definitive SILSO observations")
    definitive, study_sha = freeze_study_frame(full)
    metadata = {
        "name": "WDC-SILSO monthly mean total sunspot number Version 2.0",
        "url": DATA_URL,
        "doi": DATA_DOI,
        "license": DATA_LICENSE,
        "source": source,
        "source_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "study_input_sha256": study_sha,
        "study_cutoff_month": STUDY_CUTOFF_MONTH,
        "n_months_raw_nonmissing": int(len(full)),
        "n_months_definitive_available": int(len(available_definitive)),
        "n_months_study": int(len(definitive)),
        "provisional_rows_available": int((full["flag"] == 0).sum()),
        "post_cutoff_rows_excluded": int((full["date"] > pd.Timestamp(f"{STUDY_CUTOFF_MONTH}-01")).sum()),
        "first_month": definitive["date"].iloc[0].strftime("%Y-%m"),
        "last_study_month": definitive["date"].iloc[-1].strftime("%Y-%m"),
        "last_definitive_available_month": available_definitive["date"].iloc[-1].strftime("%Y-%m"),
        "last_available_month": full["date"].iloc[-1].strftime("%Y-%m"),
    }
    return definitive, metadata


def make_windows(values, window: int, horizon: int):
    values = np.asarray(values, dtype="float32")
    n = len(values) - window - horizon + 1
    if n <= 0:
        raise ValueError("not enough observations to construct windows")
    X = np.asarray([values[i : i + window] for i in range(n)], dtype="float32")
    y = np.asarray([values[i + window + horizon - 1] for i in range(n)], dtype="float32")
    target_indices = np.arange(window + horizon - 1, len(values), dtype=int)
    return X, y, target_indices


def fixed_target_boundaries(n_values: int, test_fraction: float = .20, validation_fraction_of_dev: float = .15):
    test_target_start = int((1.0 - test_fraction) * n_values)
    validation_target_start = int((1.0 - validation_fraction_of_dev) * test_target_start)
    if not 0 < validation_target_start < test_target_start < n_values:
        raise ValueError("invalid frozen target-date boundaries")
    return validation_target_start, test_target_start


def chronological_split(
    values,
    window: int = WINDOW,
    horizon: int = 1,
    validation_target_start: int | None = None,
    test_target_start: int | None = None,
):
    raw_X, raw_y, target_indices = make_windows(values, window, horizon)
    if len(raw_X) < 100:
        raise ValueError("not enough windows for frozen forecasting protocol")
    if validation_target_start is None or test_target_start is None:
        validation_target_start, test_target_start = fixed_target_boundaries(len(values))

    train_mask = target_indices < validation_target_start
    validation_mask = (target_indices >= validation_target_start) & (target_indices < test_target_start)
    test_mask = target_indices >= test_target_start
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("target-date boundaries produce an empty forecasting partition")

    last_training_target = int(target_indices[train_mask][-1])
    fit_values = np.asarray(values[: last_training_target + 1], dtype="float32")
    mean = float(fit_values.mean())
    std = float(fit_values.std())
    if not np.isfinite(std) or std == 0:
        raise ValueError("training-era standard deviation is zero or non-finite")

    X = (raw_X - mean) / std
    y = (raw_y - mean) / std
    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val": X[validation_mask],
        "y_val": y[validation_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "target_indices_test": target_indices[test_mask],
        "mean": mean,
        "std": std,
        "validation_target_start": int(validation_target_start),
        "test_target_start": int(test_target_start),
        "last_training_target_index": last_training_target,
    }


def error_metrics(observed, forecast) -> dict:
    observed = np.asarray(observed, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean((forecast - observed) ** 2))),
        "mae": float(np.mean(np.abs(forecast - observed))),
    }


def select_ridge_baseline(X_train, y_train, X_val, y_val):
    candidates = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        candidates.append((val_mse, alpha, model))
    candidates.sort(key=lambda x: (x[0], x[1]))
    best_mse, best_alpha, best_model = candidates[0]
    return best_model, {
        "selected_alpha": float(best_alpha),
        "validation_mse_scaled": float(best_mse),
        "candidate_alphas": [float(a) for a in RIDGE_ALPHAS],
    }


def select_hgb_baseline(X_train, y_train, X_val, y_val):
    candidates = []
    for config in HGB_CANDIDATES:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=300,
            learning_rate=config["learning_rate"],
            max_leaf_nodes=config["max_leaf_nodes"],
            l2_regularization=1.0,
            early_stopping=False,
            random_state=PRIMARY_SEED,
        ).fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        candidates.append((val_mse, config, model))
    candidates.sort(key=lambda x: (x[0], x[1]["learning_rate"], x[1]["max_leaf_nodes"]))
    best_mse, best_config, best_model = candidates[0]
    return best_model, {
        "selected_learning_rate": float(best_config["learning_rate"]),
        "selected_max_leaf_nodes": int(best_config["max_leaf_nodes"]),
        "max_iter": 300,
        "early_stopping": False,
        "validation_mse_scaled": float(best_mse),
        "candidate_grid": [dict(c) for c in HGB_CANDIDATES],
    }


def seasonal_naive_from_context(X_scaled, mean: float, std: float, horizon: int):
    if not 1 <= horizon <= 12:
        raise ValueError("seasonal baseline supports horizons 1..12")
    position = X_scaled.shape[1] + horizon - 13
    return X_scaled[:, position] * std + mean


def activity_error_analysis(observed, forecasts: dict[str, np.ndarray], train_values) -> dict:
    threshold = float(np.quantile(np.asarray(train_values, dtype=float), .75))
    high = np.asarray(observed) >= threshold
    out = {"training_75th_percentile": threshold, "high_activity_test_points": int(high.sum())}
    for name, pred in forecasts.items():
        pred = np.asarray(pred, dtype=float)
        obs = np.asarray(observed, dtype=float)
        out[name] = {
            "mae_high_activity": float(np.mean(np.abs(pred[high] - obs[high]))) if high.any() else None,
            "mae_other": float(np.mean(np.abs(pred[~high] - obs[~high]))) if (~high).any() else None,
        }
    return out


def moving_block_bootstrap_mae_delta(observed, forecast_a, forecast_b, block: int = 12, n_boot: int = 2000, seed: int = 20260925):
    observed = np.asarray(observed, dtype=float)
    loss_delta = np.abs(np.asarray(forecast_a) - observed) - np.abs(np.asarray(forecast_b) - observed)
    n = len(loss_delta)
    if n < block:
        raise ValueError("test sequence shorter than bootstrap block")
    starts = np.arange(0, n - block + 1)
    rng = np.random.default_rng(seed)
    means = []
    blocks_needed = int(np.ceil(n / block))
    for _ in range(n_boot):
        sample = np.concatenate([loss_delta[s : s + block] for s in rng.choice(starts, size=blocks_needed, replace=True)])[:n]
        means.append(float(sample.mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_mae_delta_a_minus_b": float(loss_delta.mean()),
        "moving_block_bootstrap_95_interval": [float(lo), float(hi)],
        "block_months": int(block),
        "interpretation": "Negative values favor forecast A. Interval is a serial-dependence-aware descriptive uncertainty summary, not a universal significance claim.",
    }


def train_transformer(X_train, y_train, X_val, y_val, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, seed: int = PRIMARY_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    model = TransformerForecaster(window=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    loss_fn = nn.MSELoss()
    best_state, best_val, best_epoch, stale = None, float("inf"), 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(torch.tensor(X_val)), torch.tensor(y_val)).item())
        history.append({"epoch": epoch, "train_mse": float(np.mean(batch_losses)), "val_mse": val_loss})
        if val_loss < best_val - 1e-7:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def fit_horizon(values, dates, horizon: int, boundaries: tuple[int, int], max_epochs: int, quick: bool = False):
    split = chronological_split(
        values, window=WINDOW, horizon=horizon,
        validation_target_start=boundaries[0], test_target_start=boundaries[1]
    )
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]
    mean, std = split["mean"], split["std"]
    observed = y_test * std + mean

    forecasts = {
        "persistence": X_test[:, -1] * std + mean,
        "seasonal_naive": seasonal_naive_from_context(X_test, mean, std, horizon),
    }
    ridge, ridge_selection = select_ridge_baseline(X_train, y_train, X_val, y_val)
    forecasts["ridge"] = ridge.predict(X_test) * std + mean

    hgb, hgb_selection = select_hgb_baseline(X_train, y_train, X_val, y_val)
    forecasts["hist_gradient_boosting"] = hgb.predict(X_test) * std + mean

    transformer_seeds = (PRIMARY_SEED,) if quick else TRANSFORMER_SEEDS
    transformer_runs, transformer_predictions, transformer_primary, primary_history = {}, {}, None, []
    for seed in transformer_seeds:
        model, history, best_epoch = train_transformer(
            X_train, y_train, X_val, y_val, max_epochs=max_epochs, seed=seed
        )
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_test)).cpu().numpy() * std + mean
        transformer_runs[str(seed)] = {"metrics": error_metrics(observed, pred), "best_epoch": int(best_epoch)}
        transformer_predictions[str(seed)] = pred
        if seed == PRIMARY_SEED:
            transformer_primary = pred
            primary_history = history
    forecasts["transformer_seed_42"] = transformer_primary

    transformer_mae = np.asarray([r["metrics"]["mae"] for r in transformer_runs.values()], dtype=float)
    transformer_rmse = np.asarray([r["metrics"]["rmse"] for r in transformer_runs.values()], dtype=float)
    metrics = {name: error_metrics(observed, pred) for name, pred in forecasts.items()}
    metrics["transformer_repeated_seed_summary"] = {
        "mae_mean": float(transformer_mae.mean()),
        "mae_std": float(transformer_mae.std(ddof=1)) if len(transformer_mae) > 1 else 0.0,
        "rmse_mean": float(transformer_rmse.mean()),
        "rmse_std": float(transformer_rmse.std(ddof=1)) if len(transformer_rmse) > 1 else 0.0,
        "runs": transformer_runs,
    }

    train_raw_end = split["last_training_target_index"] + 1
    uncertainty = {
        baseline: moving_block_bootstrap_mae_delta(observed, transformer_primary, forecasts[baseline])
        for baseline in ("persistence", "seasonal_naive", "ridge", "hist_gradient_boosting")
    }
    uncertainty_all_seeds = {}
    for baseline in ("persistence", "seasonal_naive", "ridge", "hist_gradient_boosting"):
        per_seed = {
            seed: moving_block_bootstrap_mae_delta(
                observed, pred, forecasts[baseline], seed=20260925 + int(seed)
            )
            for seed, pred in transformer_predictions.items()
        }
        seed_deltas = np.asarray(
            [d["mean_mae_delta_a_minus_b"] for d in per_seed.values()], dtype=float
        )
        uncertainty_all_seeds[baseline] = {
            "per_seed": per_seed,
            "mean_delta_across_seeds": float(seed_deltas.mean()),
            "std_delta_across_seeds": float(seed_deltas.std(ddof=1)) if len(seed_deltas) > 1 else 0.0,
            "seeds_favoring_transformer": int((seed_deltas < 0).sum()),
            "n_seeds": int(len(seed_deltas)),
            "interpretation": "Seed-level paired block-bootstrap summaries plus observed across-seed delta spread; descriptive robustness, not independent-replication inference.",
        }
    activity = activity_error_analysis(observed, forecasts, values[:train_raw_end])
    target_dates = pd.Series(dates).iloc[split["target_indices_test"]].dt.strftime("%Y-%m").tolist()
    midpoint = len(observed) // 2
    era = {
        "early_test": {name: error_metrics(observed[:midpoint], pred[:midpoint]) for name, pred in forecasts.items()},
        "late_test": {name: error_metrics(observed[midpoint:], pred[midpoint:]) for name, pred in forecasts.items()},
    }
    return {
        "horizon_months": int(horizon),
        "train_windows": int(len(X_train)),
        "validation_windows": int(len(X_val)),
        "test_windows": int(len(X_test)),
        "metrics": metrics,
        "baseline_selection": {
            "ridge": ridge_selection,
            "hist_gradient_boosting": hgb_selection,
        },
        "transformer_vs_baseline_uncertainty": uncertainty,
        "transformer_vs_baseline_uncertainty_all_seeds": uncertainty_all_seeds,
        "activity_error_analysis": activity,
        "era_robustness": era,
        "training_history_seed_42": primary_history,
        "plot": {
            "dates": target_dates,
            "observed": observed.astype(float).tolist(),
            "forecasts": {k: np.asarray(v, dtype=float).tolist() for k, v in forecasts.items()},
        },
    }


def context_sensitivity(values, boundaries: tuple[int, int], horizon: int = 1, max_epochs: int = MAX_EPOCHS):
    out = {}
    for window in (60, 132, 264):
        split = chronological_split(
            values, window=window, horizon=horizon,
            validation_target_start=boundaries[0], test_target_start=boundaries[1]
        )
        X_train, y_train = split["X_train"], split["y_train"]
        X_val, y_val = split["X_val"], split["y_val"]
        X_test, y_test = split["X_test"], split["y_test"]
        mean, std = split["mean"], split["std"]
        observed = y_test * std + mean
        ridge, ridge_selection = select_ridge_baseline(X_train, y_train, X_val, y_val)
        ridge_pred = ridge.predict(X_test) * std + mean
        model, _, best_epoch = train_transformer(X_train, y_train, X_val, y_val, max_epochs=max_epochs, seed=PRIMARY_SEED)
        model.eval()
        with torch.no_grad():
            t_pred = model(torch.tensor(X_test)).cpu().numpy() * std + mean
        out[str(window)] = {
            "transformer_max_epochs": int(max_epochs),
            "ridge": error_metrics(observed, ridge_pred),
            "ridge_selected_alpha": ridge_selection["selected_alpha"],
            "transformer_seed_42": error_metrics(observed, t_pred),
            "transformer_best_epoch": int(best_epoch),
        }
    return out


def write_figures(horizons: dict, results_dir: Path) -> None:
    figdir = results_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for key, result in horizons.items():
        plot = result["plot"]
        dates = pd.to_datetime(plot["dates"])
        observed = np.asarray(plot["observed"])
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(dates, observed, label="observed", linewidth=1.4)
        for name in ("seasonal_naive", "ridge", "hist_gradient_boosting", "transformer_seed_42"):
            ax.plot(dates, plot["forecasts"][name], label=name, linewidth=.9, alpha=.85)
        ax.set_title(f"SILSO monthly sunspot forecast — {key}-month horizon")
        ax.set_ylabel("Monthly mean total sunspot number")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(figdir / f"forecast_horizon_{key}.png", dpi=170)
        plt.close(fig)


def _strip_plot(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "plot"}


def build_results_latex(results: dict) -> str:
    bs = "\\"
    row_end = bs + bs
    lines = [
        f"{bs}section{{Generated empirical results}}",
        "This section is generated by \\texttt{src/run\_experiment.py}; numerical values should not be hand-edited.",
        "",
        f"Frozen study input: {results['dataset']['first_month']} through "
        f"{results['dataset']['last_study_month']}; SHA-256 "
        f"\\texttt{{{results['dataset']['study_input_sha256']}}}.",
        "",
        f"{bs}begin{{table}}[htbp]",
        f"{bs}centering",
        f"{bs}small",
        f"{bs}begin{{tabular}}{{rrrrrr}}",
        f"{bs}toprule",
        "Horizon & Persistence MAE & Seasonal MAE & Ridge MAE & HGB MAE & Transformer MAE " + row_end,
        f"{bs}midrule",
    ]
    for horizon, row in results["horizons"].items():
        m = row["metrics"]
        lines.append(
            f"{horizon} & {m['persistence']['mae']:.3f} & {m['seasonal_naive']['mae']:.3f} & "
            f"{m['ridge']['mae']:.3f} & {m['hist_gradient_boosting']['mae']:.3f} & "
            f"{m['transformer_seed_42']['mae']:.3f} " + row_end
        )
    lines += [
        f"{bs}bottomrule",
        f"{bs}end{{tabular}}",
        f"{bs}caption{{Primary seed-42 mean absolute error under common target-date boundaries.}}",
        f"{bs}label{{tab:primary-results}}",
        f"{bs}end{{table}}",
        "",
        f"{bs}paragraph{{Boundary integrity.}}",
        f"Validation targets begin {results['protocol']['validation_target_start_month']} and test targets begin "
        f"{results['protocol']['test_target_start_month']} for every primary horizon and context-sensitivity condition.",
        "",
        f"{bs}begin{{table}}[htbp]",
        f"{bs}centering",
        f"{bs}small",
        f"{bs}begin{{tabular}}{{rrr}}",
        f"{bs}toprule",
        "Horizon & Comparator & Transformer--baseline $\\Delta$MAE [95\\% block interval] " + row_end,
        f"{bs}midrule",
    ]
    comparator_labels = {
        "persistence": "Persistence",
        "seasonal_naive": "Seasonal naive",
        "ridge": "Ridge",
        "hist_gradient_boosting": "HGB",
    }
    for horizon, row in results["horizons"].items():
        for key in ("persistence", "seasonal_naive", "ridge", "hist_gradient_boosting"):
            d = row["transformer_vs_baseline_uncertainty"][key]
            lo, hi = d["moving_block_bootstrap_95_interval"]
            lines.append(
                f"{horizon} & {comparator_labels[key]} & "
                f"{d['mean_mae_delta_a_minus_b']:.3f} [{lo:.3f}, {hi:.3f}] " + row_end
            )
    lines += [
        f"{bs}bottomrule",
        f"{bs}end{{tabular}}",
        f"{bs}caption{{Paired seed-42 absolute-error differences against every declared primary baseline. Negative values favor the Transformer. Intervals use 12-month moving blocks.}}",
        f"{bs}end{{table}}",
        "",
        f"{bs}begin{{table}}[htbp]",
        f"{bs}centering",
        f"{bs}small",
        f"{bs}begin{{tabular}}{{rrr}}",
        f"{bs}toprule",
        "Horizon & Comparator & Mean $\\Delta$MAE across Transformer seeds " + row_end,
        f"{bs}midrule",
    ]
    for horizon, row in results["horizons"].items():
        for key in ("persistence", "seasonal_naive", "ridge", "hist_gradient_boosting"):
            d = row["transformer_vs_baseline_uncertainty_all_seeds"][key]
            lines.append(
                f"{horizon} & {comparator_labels[key]} & "
                f"{d['mean_delta_across_seeds']:.3f} $\\pm$ {d['std_delta_across_seeds']:.3f} "
                f"({d['seeds_favoring_transformer']}/{d['n_seeds']} seeds favor Transformer) " + row_end
            )
    lines += [
        f"{bs}bottomrule",
        f"{bs}end{{tabular}}",
        f"{bs}caption{{Optimization-seed robustness of Transformer-minus-baseline MAE.}}",
        f"{bs}end{{table}}",
        "",
        f"{bs}begin{{table}}[htbp]",
        f"{bs}centering",
        f"{bs}small",
        f"{bs}begin{{tabular}}{{rrrrr}}",
        f"{bs}toprule",
        "Context & Ridge $\\alpha$ & Ridge MAE & Transformer MAE & Transformer best epoch " + row_end,
        f"{bs}midrule",
    ]
    context = results.get("context_sensitivity_horizon_1", {})
    if "status" not in context:
        for window, row in context.items():
            lines.append(
                f"{window} & {row['ridge_selected_alpha']:.1f} & {row['ridge']['mae']:.3f} & "
                f"{row['transformer_seed_42']['mae']:.3f} & {row['transformer_best_epoch']} " + row_end
            )
    lines += [
        f"{bs}bottomrule",
        f"{bs}end{{tabular}}",
        f"{bs}caption{{Horizon-1 context sensitivity on identical target dates, invariant trainable Transformer size, and the same maximum epoch budget as the primary study.}}",
        f"{bs}end{{table}}",
        "",
    ]
    return "\n".join(lines)


def write_summary(results: dict, path: Path) -> None:
    lines = [
        "# Empirical Results Summary", "",
        "Generated by src/run_experiment.py; numerical values should not be hand-edited.", "",
        f"Source: WDC-SILSO Version 2.0, frozen study observations {results['dataset']['first_month']} through {results['dataset']['last_study_month']}.", "",
        f"Study-input SHA-256: \`{results['dataset']['study_input_sha256']}\`.", "",
        f"Common validation target start: {results['protocol']['validation_target_start_month']}; common test target start: {results['protocol']['test_target_start_month']}.", "",
        "| Horizon | Persistence MAE | Seasonal MAE | Ridge MAE | HGB MAE | Transformer MAE (seed 42) | Transformer repeated-seed mean ± SD |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, row in results["horizons"].items():
        m = row["metrics"]
        tr = m["transformer_repeated_seed_summary"]
        lines.append(
            f"| {horizon} | {m['persistence']['mae']:.3f} | {m['seasonal_naive']['mae']:.3f} | "
            f"{m['ridge']['mae']:.3f} | {m['hist_gradient_boosting']['mae']:.3f} | "
            f"{m['transformer_seed_42']['mae']:.3f} | {tr['mae_mean']:.3f} ± {tr['mae_std']:.3f} |"
        )
    lines += ["", "## Validation-selected baseline settings", "",
        "| Horizon | Ridge alpha | HGB learning rate | HGB max leaf nodes |",
        "|---:|---:|---:|---:|",
    ]
    for horizon, row in results["horizons"].items():
        sel = row["baseline_selection"]
        lines.append(
            f"| {horizon} | {sel['ridge']['selected_alpha']:.1f} | "
            f"{sel['hist_gradient_boosting']['selected_learning_rate']:.2f} | "
            f"{sel['hist_gradient_boosting']['selected_max_leaf_nodes']} |"
        )

    lines += ["", "## Paired Transformer-vs-baseline MAE differences", "",
        "Negative values favor the seed-42 Transformer. Intervals use 12-month moving blocks.", "",
        "| Horizon | vs persistence | vs seasonal naive | vs Ridge | vs HGB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon, row in results["horizons"].items():
        u = row["transformer_vs_baseline_uncertainty"]
        cells = []
        for name in ("persistence", "seasonal_naive", "ridge", "hist_gradient_boosting"):
            d = u[name]
            lo, hi = d["moving_block_bootstrap_95_interval"]
            cells.append(f"{d['mean_mae_delta_a_minus_b']:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {horizon} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    lines += ["", "## All-seed Transformer-vs-baseline robustness", "",
        "Negative mean deltas favor the Transformer; the seed-count column shows how often that direction appears across seeds.", "",
        "| Horizon | Comparator | Mean MAE delta across seeds | SD across seeds | Seeds favoring Transformer |",
        "|---:|---|---:|---:|---:|",
    ]
    for horizon, row in results["horizons"].items():
        u = row["transformer_vs_baseline_uncertainty_all_seeds"]
        for name in ("persistence", "seasonal_naive", "ridge", "hist_gradient_boosting"):
            d = u[name]
            lines.append(
                f"| {horizon} | {name} | {d['mean_delta_across_seeds']:.3f} | "
                f"{d['std_delta_across_seeds']:.3f} | {d['seeds_favoring_transformer']}/{d['n_seeds']} |"
            )

    lines += ["", "## Activity and test-era checks", "",
        "| Horizon | Transformer high-activity MAE | HGB high-activity MAE | Transformer early MAE | Transformer late MAE | HGB early MAE | HGB late MAE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, row in results["horizons"].items():
        a = row["activity_error_analysis"]
        e = row["era_robustness"]
        lines.append(
            f"| {horizon} | {a['transformer_seed_42']['mae_high_activity']:.3f} | "
            f"{a['hist_gradient_boosting']['mae_high_activity']:.3f} | "
            f"{e['early_test']['transformer_seed_42']['mae']:.3f} | "
            f"{e['late_test']['transformer_seed_42']['mae']:.3f} | "
            f"{e['early_test']['hist_gradient_boosting']['mae']:.3f} | "
            f"{e['late_test']['hist_gradient_boosting']['mae']:.3f} |"
        )

    context = results.get("context_sensitivity_horizon_1", {})
    if "status" not in context:
        lines += ["", "## Horizon-1 context sensitivity", "",
            "| Context months | Ridge alpha | Ridge MAE | Transformer MAE | Transformer best epoch |",
            "|---:|---:|---:|---:|---:|",
        ]
        for window, row in context.items():
            lines.append(
                f"| {window} | {row['ridge_selected_alpha']:.1f} | {row['ridge']['mae']:.3f} | "
                f"{row['transformer_seed_42']['mae']:.3f} | {row['transformer_best_epoch']} |"
            )

    lines += [
        "", "## Interpretation guardrail", "",
        "The Transformer is not assumed to win. Moving-block bootstrap intervals compare paired absolute errors while preserving local serial dependence more than an iid bootstrap. The 12-month seasonal-naive forecast equals persistence at a 12-month horizon by construction. Results remain specific to this frozen series, forecast horizon, context length and evaluation era.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(results_dir: str | Path = "results", data_path: str | Path | None = None, quick: bool = False, max_epochs: int = MAX_EPOCHS):
    frame, metadata = load_real_series(data_path=data_path)
    values = frame["sunspots"].to_numpy(dtype="float32")
    dates = frame["date"]
    boundaries = fixed_target_boundaries(len(values))
    horizons_with_plot = {
        str(h): fit_horizon(values, dates, h, boundaries=boundaries, max_epochs=max_epochs, quick=quick)
        for h in HORIZONS
    }
    results = {
        "research_bundle": True,
        "status": "quick_smoke_run" if quick else "complete",
        "dataset": metadata,
        "protocol": {
            "context_months": WINDOW,
            "horizons_months": list(HORIZONS),
            "test_fraction": .20,
            "validation_fraction_of_development": .15,
            "transformer_seeds": [PRIMARY_SEED] if quick else list(TRANSFORMER_SEEDS),
            "max_epochs": int(max_epochs),
            "patience": PATIENCE,
            "definitive_only": True,
            "study_cutoff_month": STUDY_CUTOFF_MONTH,
            "study_input_sha256": EXPECTED_STUDY_INPUT_SHA256,
            "validation_target_start_index": int(boundaries[0]),
            "test_target_start_index": int(boundaries[1]),
            "validation_target_start_month": dates.iloc[boundaries[0]].strftime("%Y-%m"),
            "test_target_start_month": dates.iloc[boundaries[1]].strftime("%Y-%m"),
            "shared_target_boundaries_across_horizons_and_contexts": True,
            "evaluation_mode": "rolling_origin_direct_forecast_with_observed_history",
            "positional_encoding": "fixed_sinusoidal",
            "baseline_selection": "chronological_validation_mse",
            "ridge_alpha_candidates": list(RIDGE_ALPHAS),
            "hgb_candidate_grid": [dict(c) for c in HGB_CANDIDATES],
            "hgb_internal_early_stopping": False,
        },
        "horizons": {k: _strip_plot(v) for k, v in horizons_with_plot.items()},
        "context_sensitivity_horizon_1": {"status": "skipped_in_quick_mode"} if quick else context_sensitivity(values, boundaries, max_epochs=max_epochs),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results, out / "summary.md")
    write_figures(horizons_with_plot, out)
    Path("paper").mkdir(exist_ok=True)
    Path("paper/results.md").write_text(
        "# Results\n\n" + (out / "summary.md").read_text(encoding="utf-8").replace("# Empirical Results Summary\n\n", "", 1),
        encoding="utf-8",
    )
    Path("paper/results.tex").write_text(build_results_latex(results), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SILSO Transformer forecasting research bundle")
    parser.add_argument("--data-path", default=None, help="Optional local SN_m_tot_V2.0.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="One Transformer seed and no context sensitivity")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    result = run_experiment(args.results_dir, args.data_path, args.quick, args.max_epochs)
    print(json.dumps({"status": result["status"], "dataset": result["dataset"]}, indent=2))


if __name__ == "__main__":
    main()
