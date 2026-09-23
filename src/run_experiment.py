from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.datasets import sunspots

torch.manual_seed(42)
torch.set_num_threads(1)

data = sunspots.load_pandas().data
values = data["SUNACTIVITY"].to_numpy(dtype="float32")
years = data["YEAR"].to_numpy()

window = 24
n_windows = len(values) - window
split = int(0.8 * n_windows)

# Fit normalization only on values that belong to the training period.
train_end = split + window
train_mean = float(values[:train_end].mean())
train_std = float(values[:train_end].std())
scaled = (values - train_mean) / train_std

X = np.array(
    [scaled[i:i + window] for i in range(n_windows)],
    dtype="float32",
)
y = np.array(
    [scaled[i + window] for i in range(n_windows)],
    dtype="float32",
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split])),
    batch_size=32,
    shuffle=False,
)

class TransformerForecaster(nn.Module):
    def __init__(self):
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

model = TransformerForecaster()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
loss_fn = nn.MSELoss()

losses = []
for _ in range(24):
    batch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    losses.append(float(np.mean(batch_losses)))

with torch.no_grad():
    forecast = model(torch.tensor(X[split:])).numpy()
    forecast = forecast * train_std + train_mean
    observed = y[split:] * train_std + train_mean

results = {
    "rmse": float(np.sqrt(np.mean((forecast - observed) ** 2))),
    "mae": float(np.mean(np.abs(forecast - observed))),
    "window_years": window,
    "n_windows": int(n_windows),
    "train_windows": int(split),
    "test_windows": int(n_windows - split),
}

Path("results").mkdir(exist_ok=True)
Path("results/metrics.json").write_text(json.dumps(results, indent=2))

plt.figure(figsize=(9, 5))
plt.plot(years, values)
plt.xlabel("Year")
plt.ylabel("Sunspot activity")
plt.title("Historical sunspot series")
plt.tight_layout()
plt.savefig("assets/03_data_or_model.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(observed, label="observed")
plt.plot(forecast, label="forecast")
plt.xlabel("Held-out step")
plt.ylabel("Sunspot activity")
plt.title("Transformer chronological forecast")
plt.legend()
plt.tight_layout()
plt.savefig("assets/04_evaluation_or_results.png", dpi=150)
plt.close()

print(json.dumps(results, indent=2))
