"""Summarize the registered moment and injection measurements and render main figure."""
import csv
import json
from pathlib import Path
import argparse

import numpy as np

from phase2_cumulative_moments import write_csv


def load(path):
    return np.genfromtxt(path,delimiter=",",names=True,dtype=None,encoding="utf8")


def select(t,group):
    if group=="SDSS": return t[t["survey"]=="SDSS"]
    t=t[t["survey"]=="JWST"]
    if group=="JWST40": return t
    t=t[t["snr"]>=15]
    if group=="JWST unflagged":
        t=t[~t["abs_flag"]]
    return t


def scatter(h,log_width):
    hc=h-np.mean(h,axis=-1,keepdims=True)
    yc=log_width-np.mean(log_width,axis=-1,keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope=np.sum(hc*yc,axis=-1,keepdims=True)/np.sum(hc**2,axis=-1,keepdims=True)
    return np.std(yc-slope*hc,axis=-1,ddof=1)


def bootstrap(h,y,z,rng):
    draws=rng.integers(len(h),size=(5000,len(h)))
    first=scatter(h[draws],y[draws])
    delta=first-scatter(h[draws],z[draws])
    return np.nanquantile(first,[.025,.975]),np.nanquantile(delta,[.025,.975])


def observed(measured,group,windows,rng):
    rows=[]
    measured=select(measured,group)
    ref={r["source"]:r["sigma"] for r in measured[measured["V"]==2500]}
    for w in windows:
        t=measured[(measured["V"]==w)&np.isfinite(measured["sigma"])]
        h=t["h"]
        y=np.log10(t["sigma"])
        width=scatter(h,y)
        z=np.log10([ref[r] for r in t["source"]])
        good=np.isfinite(z)
        ci,_=bootstrap(h,y,y,rng)
        _,dci=bootstrap(h[good],y[good],z[good],rng)
        delta=scatter(h[good],y[good])-scatter(h[good],z[good])
        rows.append(dict(group=group,V=int(w),n=len(y),paired_n=int(good.sum()),s=width,lo=ci[0],hi=ci[1],
                         delta=delta,delta_lo=dci[0],delta_hi=dci[1],s_single_gauss=scatter(h,np.log10(t["sg"])),
                         median_sigma=np.median(t["sigma"])))
    return rows


def calibration(table,groups=("JWST","SDSS","JWST40")):
    rows=[]
    for name in groups:
        t=select(table,name)
        for w in np.unique(t["V"]):
            for k in np.unique(t["profile"]):
                data=t[(t["V"]==w)&(t["profile"]==k)]
                err=np.log10(data["recovered"]/data["truth"])
                valid=np.isfinite(err)
                bias=np.nanmedian(err)
                q=np.nanquantile(abs(err),.68)
                rows.append(dict(group=name,V=int(w),profile=k,usable=valid.mean(),bias=bias,error68=q,
                                 passed=bool(valid.mean()>=.85 and abs(bias)<=.03 and q<=.08)))
    return rows


def predictive(measured,mocks,group,rng):
    # Compare each null to the exact observed paired-source set. Each source draws one jointly
    # recovered realization at the two windows. Includes the shared measurement noise across windows.
    m=select(measured,group)
    t=select(mocks,group)
    first={r["source"]:r for r in m[(m["V"]==2500)&np.isfinite(m["sigma"])]}
    second={r["source"]:r for r in m[(m["V"]==4500)&np.isfinite(m["sigma"])]}
    ids=sorted(first.keys() & second.keys())
    h=np.array([first[i]["h"] for i in ids])
    z,y=[np.log10([q[i]["sigma"] for i in ids]) for q in (first,second)]
    obs=scatter(h,y)-scatter(h,z)
    rows=[]
    for shape in np.unique(t["profile"]):
        pop=t[t["profile"]==shape]
        inner=[];outer=[];true0=[];true1=[];full=[]
        available=[]
        for key in ids:
            part=pop[pop["source"]==key]
            a,b=[np.sort(part[part["V"]==w],order="rep") for w in [2500,4500]]
            assert np.array_equal(a["rep"],b["rep"])
            use=np.flatnonzero(np.isfinite(a["recovered"]+b["recovered"]))
            if not len(use): continue
            index=rng.choice(use,4000,replace=True)
            available.append(key)
            for store,table,col in [(inner,a,"recovered"),(outer,b,"recovered"),(true0,a,"truth"),
                                    (true1,b,"truth"),(full,b,"full_reference")]:
                store.append(np.log10(table[col][index]))
        keep=np.isin(ids,available)
        hx=h[keep]
        si,so,ti,to,fulls=[scatter(hx,np.asarray(a).T) for a in (inner,outer,true0,true1,full)]
        delta=so-si
        q=np.quantile(delta,[.025,.5,.975])
        rows.append(dict(group=group,profile=shape,n=len(hx),obs_paired_delta=float(scatter(hx,y[keep])-scatter(hx,z[keep])),
                         predicted_delta_lo=q[0],predicted_delta_med=q[1],predicted_delta_hi=q[2],
                         pred_s2500_med=np.median(si),pred_s4500_med=np.median(so),
                         true_s2500_med=np.median(ti),true_s4500_med=np.median(to),true_full_med=np.median(fulls),
                         delta_true_med=np.median(to-ti)))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory",type=Path)
    out=ap.parse_args().directory
    rng=np.random.default_rng(221121)
    data=load(out/"measurements.csv")
    sim=load(out/"injections.csv")
    cal=calibration(sim)
    obs=[]
    for g in ["JWST","SDSS","JWST40","JWST unflagged"]:
        obs+=observed(data,g,np.unique(data["V"]),rng)
    ext=load(out/"edge_sensitivity_measurements.csv")
    sec=[r for name in ["JWST","SDSS"] for r in observed(ext,name,np.unique(ext["V"]),rng)]
    cond=load(out/"source_conditioned_injections.csv")
    pred=[r for g in ["JWST","SDSS"] for r in predictive(data,cond,g,rng)]
    for filename,rows in [("calibration_summary",cal),("observed_scatter",obs),("edge_scatter",sec),
                          ("source_conditioned_predictions",pred)]:
        write_csv(out/(filename+".csv"),rows)


if __name__=="__main__":
    main()
