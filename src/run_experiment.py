from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from statsmodels.datasets import sunspots
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
WINDOW = 24


class TransformerForecaster(nn.Module):
    def __init__(self, window: int = WINDOW):
        super().__init__()
        dim = 24
        self.projection = nn.Linear(1, dim)
        self.position = nn.Parameter(torch.randn(1, window, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=48,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        z = self.projection(x.unsqueeze(-1)) + self.position
        z = self.encoder(z)
        return self.head(z[:, -1]).squeeze(-1)


def load_series():
    data = sunspots.load_pandas().data
    return (
        data["YEAR"].to_numpy(),
        data["SUNACTIVITY"].to_numpy(dtype="float32"),
    )


def prepare_windows(values, window: int = WINDOW, train_fraction: float = 0.8):
    values = np.asarray(values, dtype="float32")
    n_windows = len(values) - window
    split = int(train_fraction * n_windows)

    train_end = split + window
    train_mean = float(values[:train_end].mean())
    train_std = float(values[:train_end].std())
    if train_std == 0:
        raise ValueError("training standard deviation is zero")

    scaled = (values - train_mean) / train_std
    X = np.array(
        [scaled[i : i + window] for i in range(n_windows)],
        dtype="float32",
    )
    y = np.array(
        [scaled[i + window] for i in range(n_windows)],
        dtype="float32",
    )
    return X, y, split, train_mean, train_std


def error_metrics(observed, forecast):
    observed = np.asarray(observed)
    forecast = np.asarray(forecast)
    return {
        "rmse": float(np.sqrt(np.mean((forecast - observed) ** 2))),
        "mae": float(np.mean(np.abs(forecast - observed))),
    }


def train_transformer(
    X_train,
    y_train,
    epochs: int = 24,
    seed: int = SEED,
):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train),
        ),
        batch_size=32,
        shuffle=True,
        generator=generator,
    )

    model = TransformerForecaster(window=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()
    losses = []

    model.train()
    for _ in range(epochs):
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))

    return model, losses


def run_experiment(
    results_dir: str | Path = "results",
    epochs: int = 24,
    seed: int = SEED,
    make_plots: bool = True,
):
    years, values = load_series()
    X, y, split, train_mean, train_std = prepare_windows(values)

    observed = y[split:] * train_std + train_mean
    persistence = X[split:, -1] * train_std + train_mean

    ridge = Ridge(alpha=1.0)
    ridge.fit(X[:split], y[:split])
    ridge_forecast = ridge.predict(X[split:]) * train_std + train_mean

    model, losses = train_transformer(
        X[:split],
        y[:split],
        epochs=epochs,
        seed=seed,
    )
    model.eval()
    with torch.no_grad():
        transformer_scaled = model(torch.tensor(X[split:])).cpu().numpy()
    transformer_forecast = transformer_scaled * train_std + train_mean

    results = {
        "seed": int(seed),
        "window_years": int(WINDOW),
        "n_windows": int(len(X)),
        "train_windows": int(split),
        "test_windows": int(len(X) - split),
        "persistence": error_metrics(observed, persistence),
        "ridge": error_metrics(observed, ridge_forecast),
        "transformer": error_metrics(observed, transformer_forecast),
    }

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    if make_plots:
        figures = output / "figures"
        figures.mkdir(parents=True, exist_ok=True)

        test_years = years[WINDOW + split :]
        plt.figure(figsize=(9, 5))
        plt.plot(test_years, observed, label="observed")
        plt.plot(test_years, ridge_forecast, label="ridge")
        plt.plot(
            test_years,
            transformer_forecast,
            label="transformer",
            alpha=0.8,
        )
        plt.xlabel("Year")
        plt.ylabel("Sunspot activity")
        plt.title("Chronological hold-out forecasts")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "forecast_comparison.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training MSE")
        plt.title("Transformer training loss")
        plt.tight_layout()
        plt.savefig(figures / "training_loss.png", dpi=150)
        plt.close()

    return results


def main() -> None:
    print(json.dumps(run_experiment(), indent=2))


if __name__ == "__main__":
    main()
