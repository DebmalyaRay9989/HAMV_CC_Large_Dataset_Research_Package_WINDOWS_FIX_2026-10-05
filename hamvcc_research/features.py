
from __future__ import annotations
import math, json
from pathlib import Path
import numpy as np
import cv2

FEATURE_NAMES=["ink_ratio","mean_intensity","std_intensity","edge_density",
"n_components_log","largest_comp_ratio","mean_comp_area_log","row_proj_entropy",
"col_proj_entropy","row_proj_peak","col_proj_peak","aspect_ratio",
"text_line_density","whitespace_ratio","color_saturation_mean","color_saturation_std"]

def load_image(path, work_size=(224,224)):
    img=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if img is None: raise IOError(f"Cannot read image: {path}")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return cv2.resize(img,tuple(work_size),interpolation=cv2.INTER_AREA)

def deskew(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    thr=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    pts=cv2.findNonZero(thr)
    if pts is None or len(pts)<20: return img
    angle=cv2.minAreaRect(pts)[-1]
    angle=-(90+angle) if angle < -45 else -angle
    if abs(angle)>12: return img
    h,w=gray.shape; M=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def preprocess(img,cfg):
    out=img
    if cfg.get("use_denoise",True):
        out=cv2.medianBlur(out,3)
    if cfg.get("use_deskew",True):
        out=deskew(out)
    if cfg.get("use_clahe",True):
        lab=cv2.cvtColor(out,cv2.COLOR_RGB2LAB); l,a,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
        out=cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2RGB)
    if cfg.get("use_binarize",False):
        g=cv2.cvtColor(out,cv2.COLOR_RGB2GRAY)
        b=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        out=cv2.cvtColor(b,cv2.COLOR_GRAY2RGB)
    return out

def handcrafted_features(img_rgb):
    img=img_rgb.astype(np.uint8); gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV); H,W=gray.shape; eps=1e-8
    thr=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    ink=float((thr>0).mean()); mean=float(gray.mean())/255.; std=float(gray.std())/255.
    edges=cv2.Canny(gray,50,150); edge=float((edges>0).mean())
    n,lab,stats,_=cv2.connectedComponentsWithStats(thr,8)
    areas=stats[1:,cv2.CC_STAT_AREA] if n>1 else np.array([0])
    nlog=float(np.log1p(len(areas)))
    largest=float(areas.max()/(areas.sum()+eps)) if len(areas) else 0.
    meanarea=float(np.log1p(areas.mean())) if len(areas) else 0.
    rp=(thr>0).sum(1).astype(np.float32); cp=(thr>0).sum(0).astype(np.float32)
    def ent(p):
        p=p/(p.sum()+eps); return float(-(p*np.log(p+eps)).sum())
    rowe,cole=ent(rp),ent(cp)
    rowp=float(rp.max()/(W+eps)); colp=float(cp.max()/(H+eps))
    ar=float(W/(H+eps)); rm=rp.mean()
    lines=int(((rp>rm)&(np.roll(rp,1)<=rm)).sum())
    lined=float(lines/(H+eps)); white=float((gray>200).mean())
    sat=hsv[:,:,1].astype(np.float32)/255.
    return np.array([ink,mean,std,edge,nlog,largest,meanarea,rowe,cole,rowp,colp,ar,lined,white,float(sat.mean()),float(sat.std())],dtype=np.float32)

def extract_handcrafted(manifest, cfg, out_npy, error_csv=None):
    n=len(manifest); dim=len(FEATURE_NAMES)
    mm=np.lib.format.open_memmap(out_npy,mode="w+",dtype=cfg.get("feature_dtype","float32"),shape=(n,dim))
    errors=[]
    for i,row in enumerate(manifest.itertuples(index=False)):
        try:
            im=preprocess(load_image(row.path,cfg["work_size"]),cfg)
            mm[i]=handcrafted_features(im)
        except Exception as e:
            mm[i]=np.nan; errors.append({"index":i,"path":row.path,"error":repr(e)})
    mm.flush()
    if error_csv is not None: __import__("pandas").DataFrame(errors).to_csv(error_csv,index=False)
    return mm, errors
