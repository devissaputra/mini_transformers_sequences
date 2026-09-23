from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from statsmodels.datasets import sunspots
torch.manual_seed(42); torch.set_num_threads(1); d=sunspots.load_pandas().data; vals=d['SUNACTIVITY'].to_numpy(dtype='float32'); years=d['YEAR'].to_numpy(); mu,sd=float(vals.mean()),float(vals.std()); z=(vals-mu)/sd; L=24; X=np.array([z[i:i+L] for i in range(len(z)-L)],dtype='float32'); y=np.array([z[i+L] for i in range(len(z)-L)],dtype='float32'); c=int(.8*len(X)); dl=DataLoader(TensorDataset(torch.tensor(X[:c]),torch.tensor(y[:c])),32,shuffle=False)
class M(nn.Module):
 def __init__(self):
  super().__init__(); dim=24; self.proj=nn.Linear(1,dim); self.pos=nn.Parameter(torch.randn(1,L,dim)*.02); layer=nn.TransformerEncoderLayer(dim,4,48,batch_first=True,dropout=.1); self.enc=nn.TransformerEncoder(layer,1); self.fc=nn.Linear(dim,1)
 def forward(self,x): z=self.proj(x.unsqueeze(-1))+self.pos; z=self.enc(z); return self.fc(z[:,-1]).squeeze(-1)
m=M(); opt=torch.optim.Adam(m.parameters(),lr=.002); lf=nn.MSELoss(); losses=[]
for _ in range(24):
 ls=[]
 for xb,yb in dl: opt.zero_grad(); loss=lf(m(xb),yb); loss.backward(); opt.step(); ls.append(loss.item())
 losses.append(float(np.mean(ls)))
with torch.no_grad(): p=m(torch.tensor(X[c:])).numpy()*sd+mu; truth=y[c:]*sd+mu
out={'rmse':float(np.sqrt(np.mean((p-truth)**2))),'mae':float(np.mean(np.abs(p-truth))),'window_years':L,'n_windows':int(len(X))}; Path('results').mkdir(exist_ok=True); Path('results/metrics.json').write_text(json.dumps(out,indent=2)); plt.figure(figsize=(9,5)); plt.plot(years,vals); plt.xlabel('Year'); plt.ylabel('Sunspot activity'); plt.title('Real historical sunspot series'); plt.tight_layout(); plt.savefig('assets/03_data_or_model.png',dpi=150); plt.close(); plt.figure(figsize=(9,5)); plt.plot(truth,label='observed'); plt.plot(p,label='forecast'); plt.xlabel('Held-out step'); plt.ylabel('Sunspot activity'); plt.title('Transformer chronological forecast'); plt.legend(); plt.tight_layout(); plt.savefig('assets/04_evaluation_or_results.png',dpi=150); plt.close(); print(json.dumps(out,indent=2))