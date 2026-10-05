from __future__ import annotations
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class ViewEncoder(nn.Module):
    def __init__(self, d, latent=64):
        super().__init__()
        h=max(64,min(256,d*4))
        self.net=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Linear(h,latent))
    def forward(self,x): return F.normalize(self.net(x),dim=1)

class ReliabilityGatedHAMVCC(nn.Module):
    """Executable research extension matching the notebook's two-view HAMV-CC design.

    Inputs are structural/handcrafted and visual views plus explicit availability masks.
    Reliability is view-local and learned; the fused embedding is a reliability-weighted
    normalized combination. A cluster head supplies the cluster-consistency term.
    """
    def __init__(self, d_struct, d_visual, latent=64, n_clusters=3):
        super().__init__()
        self.struct=ViewEncoder(d_struct,latent)
        self.visual=ViewEncoder(d_visual,latent)
        self.gate_s=nn.Sequential(nn.Linear(latent,32),nn.ReLU(),nn.Linear(32,1))
        self.gate_v=nn.Sequential(nn.Linear(latent,32),nn.ReLU(),nn.Linear(32,1))
        self.cluster_head=nn.Linear(latent,n_clusters)
    def forward(self,xs,xv,ms=None,mv=None):
        zs=self.struct(xs); zv=self.visual(xv)
        gs=self.gate_s(zs).squeeze(1); gv=self.gate_v(zv).squeeze(1)
        if ms is not None: gs=gs.masked_fill(ms<=0,-1e4)
        if mv is not None: gv=gv.masked_fill(mv<=0,-1e4)
        weights=torch.softmax(torch.stack([gs,gv],dim=1),dim=1)
        if ms is not None and mv is not None:
            both=(ms>0)&(mv>0); only_s=(ms>0)&~(mv>0); only_v=(mv>0)&~(ms>0)
            weights=torch.where(only_s[:,None],torch.tensor([1.,0.],device=weights.device),weights)
            weights=torch.where(only_v[:,None],torch.tensor([0.,1.],device=weights.device),weights)
            weights=torch.where((~both)[:,None] & (~only_s)[:,None] & (~only_v)[:,None],torch.tensor([.5,.5],device=weights.device),weights)
        fused=F.normalize(weights[:,0:1]*zs+weights[:,1:2]*zv,dim=1)
        logits=self.cluster_head(fused)
        logits_s=self.cluster_head(zs)
        logits_v=self.cluster_head(zv)
        return fused,zs,zv,weights,logits,logits_s,logits_v

def _nt_xent(z1,z2,temp=.1):
    n=z1.shape[0]; z=torch.cat([F.normalize(z1,dim=1),F.normalize(z2,dim=1)],0)
    sim=(z@z.T)/temp; mask=torch.eye(2*n,device=z.device,dtype=torch.bool)
    sim=sim.masked_fill(mask,-1e9); target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(sim,target)

def train_hamvcc(X_struct,X_visual,n_clusters,epochs=150,latent=64,batch_size=64,lr=1e-3,
                 lambda_inst=1.0,lambda_view=.5,lambda_cc=.5,use_reliability=True,
                 masks=None,seed=42,device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    xs=torch.tensor(np.asarray(X_struct,dtype=np.float32)); xv=torch.tensor(np.asarray(X_visual,dtype=np.float32))
    n=len(xs)
    ms=torch.ones(n) if masks is None or 'struct' not in masks else torch.tensor(masks['struct'],dtype=torch.float32)
    mv=torch.ones(n) if masks is None or 'visual' not in masks else torch.tensor(masks['visual'],dtype=torch.float32)
    ds=TensorDataset(xs,xv,ms,mv)
    loader=DataLoader(ds,batch_size=min(batch_size,n),shuffle=True,drop_last=False)
    model=ReliabilityGatedHAMVCC(xs.shape[1],xv.shape[1],latent,n_clusters).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    history=[]
    for ep in range(epochs):
        model.train(); losses=[]; parts=[]
        for xb,yb,mbs,mbv in loader:
            xb,yb,mbs,mbv=xb.to(device),yb.to(device),mbs.to(device),mbv.to(device)
            fused,zs,zv,w,logits,logits_s,logits_v=model(xb,yb,mbs,mbv)
            if use_reliability:
                inst=_nt_xent(fused+0.02*torch.randn_like(fused),fused+0.02*torch.randn_like(fused),.1)
            else:
                fixed=F.normalize((zs+zv)/2,dim=1)
                inst=_nt_xent(fixed+0.02*torch.randn_like(fixed),fixed+0.02*torch.randn_like(fixed),.1)
            view=1-(zs*zv).sum(1).mean()
            p=F.log_softmax(logits_s,dim=1); q=F.softmax(logits_v,dim=1).detach()
            p2=F.log_softmax(logits_v,dim=1); q2=F.softmax(logits_s,dim=1).detach()
            cc=0.5*(F.kl_div(p,q,reduction='batchmean')+F.kl_div(p2,q2,reduction='batchmean'))
            loss=lambda_inst*inst+lambda_view*view+lambda_cc*cc
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            losses.append(float(loss.detach().cpu())); parts.append((float(inst.detach().cpu()),float(view.detach().cpu()),float(cc.detach().cpu())))
        a=np.mean(parts,axis=0) if parts else [np.nan]*3
        history.append({'epoch':ep+1,'loss':float(np.mean(losses)),'instance':a[0],'cross_view':a[1],'cluster_consistency':a[2]})
    model.eval()
    with torch.inference_mode():
        fused,zs,zv,w,logits,_,_=model(xs.to(device),xv.to(device),ms.to(device),mv.to(device))
    return model,fused.cpu().numpy().astype(np.float32),w.cpu().numpy().astype(np.float32),pd.DataFrame(history)
