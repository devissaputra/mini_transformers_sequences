from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import platform
import urllib.request
from pathlib import Path

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


class TransformerForecaster(nn.Module):
    def __init__(self, window: int = WINDOW):
        super().__init__()
        dim = 32
        self.projection = nn.Linear(1, dim)
        self.position = nn.Parameter(torch.randn(1, window, dim) * 0.02)
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
    definitive = full[full["flag"] == 1].copy().reset_index(drop=True)
    if len(definitive) < 1000:
        raise ValueError("unexpectedly few definitive SILSO observations")
    metadata = {
        "name": "WDC-SILSO monthly mean total sunspot number Version 2.0",
        "url": DATA_URL,
        "doi": DATA_DOI,
        "license": DATA_LICENSE,
        "source": source,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_months_raw_nonmissing": int(len(full)),
        "n_months_definitive": int(len(definitive)),
        "provisional_rows_excluded": int(len(full) - len(definitive)),
        "first_month": definitive["date"].iloc[0].strftime("%Y-%m"),
        "last_definitive_month": definitive["date"].iloc[-1].strftime("%Y-%m"),
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


def chronological_split(values, window: int = WINDOW, horizon: int = 1, test_fraction: float = .20, validation_fraction_of_dev: float = .15):
    raw_X, raw_y, target_indices = make_windows(values, window, horizon)
    n = len(raw_X)
    if n < 100:
        raise ValueError("not enough windows for frozen forecasting protocol")
    test_start = int((1.0 - test_fraction) * n)
    val_start = int((1.0 - validation_fraction_of_dev) * test_start)
    if not 0 < val_start < test_start < n:
        raise ValueError("invalid chronological split")

    last_training_target = target_indices[val_start - 1]
    fit_values = np.asarray(values[: last_training_target + 1], dtype="float32")
    mean = float(fit_values.mean())
    std = float(fit_values.std())
    if not np.isfinite(std) or std == 0:
        raise ValueError("training-era standard deviation is zero or non-finite")

    X = (raw_X - mean) / std
    y = (raw_y - mean) / std
    return {
        "X_train": X[:val_start],
        "y_train": y[:val_start],
        "X_val": X[val_start:test_start],
        "y_val": y[val_start:test_start],
        "X_test": X[test_start:],
        "y_test": y[test_start:],
        "target_indices_test": target_indices[test_start:],
        "mean": mean,
        "std": std,
        "val_start": int(val_start),
        "test_start": int(test_start),
        "last_training_target_index": int(last_training_target),
    }


def error_metrics(observed, forecast) -> dict:
    observed = np.asarray(observed, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean((forecast - observed) ** 2))),
        "mae": float(np.mean(np.abs(forecast - observed))),
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


def fit_horizon(values, dates, horizon: int, max_epochs: int, quick: bool = False):
    split = chronological_split(values, window=WINDOW, horizon=horizon)
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]
    mean, std = split["mean"], split["std"]
    observed = y_test * std + mean

    forecasts = {
        "persistence": X_test[:, -1] * std + mean,
        "seasonal_naive": seasonal_naive_from_context(X_test, mean, std, horizon),
    }
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    forecasts["ridge"] = ridge.predict(X_test) * std + mean

    hgb = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=300, learning_rate=.05,
        max_leaf_nodes=15, l2_regularization=1.0, random_state=PRIMARY_SEED,
    ).fit(X_train, y_train)
    forecasts["hist_gradient_boosting"] = hgb.predict(X_test) * std + mean

    transformer_seeds = (PRIMARY_SEED,) if quick else TRANSFORMER_SEEDS
    transformer_runs, transformer_primary, primary_history = {}, None, []
    for seed in transformer_seeds:
        model, history, best_epoch = train_transformer(
            X_train, y_train, X_val, y_val, max_epochs=max_epochs, seed=seed
        )
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_test)).cpu().numpy() * std + mean
        transformer_runs[str(seed)] = {"metrics": error_metrics(observed, pred), "best_epoch": int(best_epoch)}
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
        for baseline in ("seasonal_naive", "ridge", "hist_gradient_boosting")
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
        "transformer_vs_baseline_uncertainty": uncertainty,
        "activity_error_analysis": activity,
        "era_robustness": era,
        "training_history_seed_42": primary_history,
        "plot": {
            "dates": target_dates,
            "observed": observed.astype(float).tolist(),
            "forecasts": {k: np.asarray(v, dtype=float).tolist() for k, v in forecasts.items()},
        },
    }


def context_sensitivity(values, horizon: int = 1, max_epochs: int = 40):
    out = {}
    for window in (60, 132, 264):
        split = chronological_split(values, window=window, horizon=horizon)
        X_train, y_train = split["X_train"], split["y_train"]
        X_val, y_val = split["X_val"], split["y_val"]
        X_test, y_test = split["X_test"], split["y_test"]
        mean, std = split["mean"], split["std"]
        observed = y_test * std + mean
        ridge = Ridge(alpha=1.0).fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test) * std + mean
        model, _, best_epoch = train_transformer(X_train, y_train, X_val, y_val, max_epochs=max_epochs, seed=PRIMARY_SEED)
        model.eval()
        with torch.no_grad():
            t_pred = model(torch.tensor(X_test)).cpu().numpy() * std + mean
        out[str(window)] = {
            "ridge": error_metrics(observed, ridge_pred),
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


def write_summary(results: dict, path: Path) -> None:
    lines = [
        "# Empirical Results Summary", "",
        "Generated by src/run_experiment.py; numerical values should not be hand-edited.", "",
        f"Source: WDC-SILSO Version 2.0, definitive observations {results['dataset']['first_month']} through {results['dataset']['last_definitive_month']}.", "",
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
    lines += [
        "", "## Interpretation guardrail", "",
        "The Transformer is not assumed to win. Moving-block bootstrap intervals compare its absolute-error difference with strong baselines while respecting serial dependence more than an iid bootstrap. Results remain specific to this series, forecast horizon, context length and evaluation era.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(results_dir: str | Path = "results", data_path: str | Path | None = None, quick: bool = False, max_epochs: int = MAX_EPOCHS):
    frame, metadata = load_real_series(data_path=data_path)
    values = frame["sunspots"].to_numpy(dtype="float32")
    dates = frame["date"]
    horizons_with_plot = {
        str(h): fit_horizon(values, dates, h, max_epochs=max_epochs, quick=quick)
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
        },
        "horizons": {k: _strip_plot(v) for k, v in horizons_with_plot.items()},
        "context_sensitivity_horizon_1": {"status": "skipped_in_quick_mode"} if quick else context_sensitivity(values),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
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
