
from __future__ import annotations
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MemmapDataset(Dataset):
    def __init__(self,X):
        self.X=X
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return np.asarray(self.X[i],dtype=np.float32)

class ProjectionEncoder(nn.Module):
    def __init__(self,d,hidden=256,proj=128):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.proj=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,proj))
    def forward(self,x):
        h=self.encoder(x); z=F.normalize(self.proj(h),dim=1); return h,z

def nt_xent(z1,z2,temp=.1):
    n=z1.shape[0]; z=torch.cat([z1,z2],0)
    sim=z@z.T/temp; mask=torch.eye(2*n,device=z.device,dtype=torch.bool); sim=sim.masked_fill(mask,-1e9)
    target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(sim,target)

def train_feature_ssl(X,epochs=30,batch_size=256,lr=1e-3,temp=.1,proj_dim=128,noise=.05,seed=42,device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if device is None: device="cuda" if torch.cuda.is_available() else "cpu"
    ds=MemmapDataset(X); loader=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=(device=="cuda"))
    model=ProjectionEncoder(X.shape[1],proj=proj_dim).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    hist=[]
    for ep in range(epochs):
        model.train(); vals=[]
        for xb in loader:
            xb=xb.to(device)
            x1=xb+noise*torch.randn_like(xb); x2=xb+noise*torch.randn_like(xb)
            _,z1=model(x1); _,z2=model(x2); loss=nt_xent(z1,z2,temp)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            vals.append(float(loss.detach().cpu()))
        hist.append({"epoch":ep+1,"loss":float(np.mean(vals))})
    # encode in deterministic batches; output is CPU/disk-sized, not GPU-sized
    model.eval(); H=np.empty((len(X),256),dtype=np.float32)
    with torch.inference_mode():
        pos=0
        for xb in DataLoader(ds,batch_size=batch_size,shuffle=False):
            h,_=model(xb.to(device)); b=len(h); H[pos:pos+b]=h.cpu().numpy(); pos+=b
    return model,H,pd.DataFrame(hist)
