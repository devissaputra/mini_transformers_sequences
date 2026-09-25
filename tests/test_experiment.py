from pathlib import Path
import numpy as np
import torch

from src.run_experiment import TransformerForecaster, parse_silso_bytes, prepare_windows

def test_repository_is_research_bundle():
    root=Path(__file__).resolve().parents[1]
    for p in ["README.md","RESEARCH_BUNDLE.md","DATA.md","REPRODUCIBILITY.md",
              "src/run_experiment.py","paper/paper.md",".github/workflows/ci.yml"]:
        assert (root/p).exists(), p

def test_silso_parser():
    raw=b"1749;01;1749.042; 96.7; -1.0; -1;0\n1749;02;1749.123; -1.0; -1.0; -1;0\n"
    frame=parse_silso_bytes(raw)
    assert len(frame)==1
    assert frame.iloc[0]["sunspots"]==96.7

def test_model_shape():
    model=TransformerForecaster(window=12)
    assert model(torch.zeros(3,12)).shape==(3,)

def test_chronological_window_partitions():
    values=np.arange(500,dtype="float32")
    X,y,val_start,test_start,mean,std=prepare_windows(values,window=24)
    assert 0 < val_start < test_start < len(X)
    assert std > 0
    assert mean == float(values[:val_start+24].mean())
