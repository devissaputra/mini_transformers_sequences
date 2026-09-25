from __future__ import annotations

import copy
import hashlib
import io
import json
import platform
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
WINDOW = 132
DATA_URL = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"

class TransformerForecaster(nn.Module):
    def __init__(self, window: int = WINDOW):
        super().__init__()
        dim = 32
        self.projection = nn.Linear(1, dim)
        self.position = nn.Parameter(torch.randn(1, window, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=64,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        z = self.projection(x.unsqueeze(-1)) + self.position
        return self.head(self.encoder(z)[:, -1]).squeeze(-1)

def parse_silso_bytes(raw: bytes):
    frame = pd.read_csv(
        io.BytesIO(raw), sep=";", header=None,
        names=["year","month","decimal_date","sunspots","std","n_obs","flag"],
        engine="python"
    )
    frame["sunspots"] = pd.to_numeric(frame["sunspots"], errors="coerce")
    frame["decimal_date"] = pd.to_numeric(frame["decimal_date"], errors="coerce")
    frame = frame.dropna(subset=["decimal_date","sunspots"])
    frame = frame[frame["sunspots"] >= 0].reset_index(drop=True)
    return frame

def load_real_series(url: str = DATA_URL):
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    frame = parse_silso_bytes(raw)
    checksum = hashlib.sha256(raw).hexdigest()
    return frame, checksum

def prepare_windows(values, window: int=WINDOW, test_fraction: float=.20, validation_fraction_of_dev: float=.15):
    values = np.asarray(values, dtype="float32")
    n_windows = len(values) - window
    if n_windows < 100:
        raise ValueError("not enough observations for frozen protocol")
    test_start = int((1.0 - test_fraction) * n_windows)
    val_start = int((1.0 - validation_fraction_of_dev) * test_start)

    scale_end = val_start + window
    mean = float(values[:scale_end].mean())
    std = float(values[:scale_end].std())
    if std == 0:
        raise ValueError("training standard deviation is zero")

    scaled = (values - mean) / std
    X = np.array([scaled[i:i+window] for i in range(n_windows)], dtype="float32")
    y = np.array([scaled[i+window] for i in range(n_windows)], dtype="float32")
    return X, y, val_start, test_start, mean, std

def error_metrics(observed, forecast):
    observed=np.asarray(observed); forecast=np.asarray(forecast)
    return {
        "rmse": float(np.sqrt(np.mean((forecast-observed)**2))),
        "mae": float(np.mean(np.abs(forecast-observed))),
    }

def train_transformer(X_train, y_train, X_val, y_val, max_epochs=60, patience=8, seed=SEED):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(
        TensorDataset(torch.tensor(X_train),torch.tensor(y_train)),
        batch_size=64,shuffle=True,generator=generator
    )
    model=TransformerForecaster(window=X_train.shape[1])
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    loss_fn=nn.MSELoss()
    best_state=None; best_val=float("inf"); best_epoch=0; stale=0
    history=[]
    for epoch in range(1,max_epochs+1):
        model.train()
        batch=[]
        for xb,yb in loader:
            optimizer.zero_grad()
            loss=loss_fn(model(xb),yb)
            loss.backward()
            optimizer.step()
            batch.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss=float(loss_fn(model(torch.tensor(X_val)),torch.tensor(y_val)).item())
        history.append({"epoch":epoch,"train_mse":float(np.mean(batch)),"val_mse":val_loss})
        if val_loss < best_val - 1e-7:
            best_val=val_loss; best_epoch=epoch; stale=0
            best_state=copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch

def run_experiment(results_dir: str | Path="results", seed: int=SEED, max_epochs: int=60):
    frame, checksum=load_real_series()
    values=frame["sunspots"].to_numpy(dtype="float32")
    X,y,val_start,test_start,mean,std=prepare_windows(values)

    X_train,y_train=X[:val_start],y[:val_start]
    X_val,y_val=X[val_start:test_start],y[val_start:test_start]
    X_test,y_test=X[test_start:],y[test_start:]

    observed=y_test*std+mean
    persistence=X_test[:,-1]*std+mean

    ridge=Ridge(alpha=1.0)
    ridge.fit(X_train,y_train)
    ridge_forecast=ridge.predict(X_test)*std+mean

    transformer,history,best_epoch=train_transformer(
        X_train,y_train,X_val,y_val,max_epochs=max_epochs,seed=seed
    )
    transformer.eval()
    with torch.no_grad():
        pred=transformer(torch.tensor(X_test)).cpu().numpy()*std+mean

    results={
        "research_bundle":True,
        "dataset":{
            "name":"WDC-SILSO monthly mean total sunspot number V2.0",
            "url":DATA_URL,"sha256":checksum,
            "n_months":int(len(frame)),
            "first_decimal_date":float(frame["decimal_date"].iloc[0]),
            "last_decimal_date":float(frame["decimal_date"].iloc[-1]),
        },
        "seed":int(seed),"window_months":WINDOW,
        "train_windows":int(len(X_train)),"validation_windows":int(len(X_val)),
        "test_windows":int(len(X_test)),"transformer_best_epoch":int(best_epoch),
        "persistence":error_metrics(observed,persistence),
        "ridge":error_metrics(observed,ridge_forecast),
        "transformer":error_metrics(observed,pred),
        "environment":{"python":platform.python_version(),"numpy":np.__version__,
                       "pandas":pd.__version__,"scikit_learn":sklearn.__version__,
                       "torch":torch.__version__},
    }
    out=Path(results_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    (out/"training_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    return results

if __name__=="__main__":
    print(json.dumps(run_experiment(),indent=2))
