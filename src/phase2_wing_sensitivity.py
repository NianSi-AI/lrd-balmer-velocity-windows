"""Secondary check for missing continuum anchors and the source-conditioned Gaussian null."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from phase2_cumulative_moments import Extraction, acquire, inputs, moments, geometry, write_csv


def process(task):
    phase,obj,seed,windows=task
    m,v,y,e,g=acquire(obj)
    ext=Extraction(m,v,e,g,windows)
    primary=ext.ready
    u=ext.v
    if not primary:
        if (ext.side & (u>0)).sum()<4:
            ext.side |= (u>=5.6)&(u<=6.3)
        if (ext.side & (u<0)).sum()<4:
            ext.side |= (u>=-6.3)&(u<=-5.4)
        ext.ready=all((ext.side & (u*sign>0)).sum()>=4 for sign in [-1,1]) and ext.inner.sum()>=18
        ext.inv=np.linalg.pinv(ext.design[ext.side]/ext.e[ext.side,None])
    y=y[g]
    base=dict(survey=m["survey"], source=m["source"],h=m["h"],sg=m["sg"],snr=m["snr"],primary_geometry=primary)
    if phase=="measure":
        (width,_),ok=ext.extract(y)
        return [dict(**base,V=int(1000*w),sigma=a,fit_ok=ok) for w,a in zip(windows,width)]
    rng=np.random.default_rng(seed)
    rows=[]
    nt=60 if m["survey"]=="JWST" else 12
    grid=np.arange(-50,50.001,.01)
    truth_mat=geometry(grid,np.ones(len(grid),bool),np.r_[windows,50.])
    if ext.ready:
        bank=ext.noise_bank(y)
    for i in range(nt):
        draw=[0.,rng.random(),rng.normal(),rng.random(),rng.uniform(.1,1.5),rng.uniform(.05,1),rng.normal()]
        for kind in ("Gaussian","variable wings"):
            if ext.ready:
                injected,func=ext.simulate(rng,y,bank,kind,draw,core_fwhm=m["sg"]/1000)
                (width,_),_=ext.extract(injected)
                truth=moments(truth_mat,func(grid),True)
            else:
                width=np.full(len(windows),np.nan)
                truth=np.full(len(windows)+1,np.nan)
            for j,w in enumerate(windows):
                rows.append(dict(**base,V=int(1000*w),rep=i,profile=kind,recovered=width[j],
                                 truth=truth[j],full_reference=truth[-1]))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["inject","measure"])
    parser.add_argument("directory",type=Path)
    a=parser.parse_args()
    cfg=json.loads((a.directory/"sensitivity_protocol.json").read_text())
    windows=np.array(cfg["windows_kms"])/1000
    if a.phase=="measure":
        assert (a.directory/"source_conditioned_injections.csv").is_file()
    tasks=[(a.phase,obj,120260001+i,windows) for i,obj in enumerate(inputs())
           if obj[1]["survey"]=="SDSS" or float(obj[1]["snr"])>=15]
    assert len(tasks)==207
    result=[]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for i,rows in enumerate(pool.map(process,tasks),1):
            result.extend(rows)
            if i%50==0: print(i,flush=True)
    name="source_conditioned_injections.csv" if a.phase=="inject" else "edge_sensitivity_measurements.csv"
    write_csv(a.directory/name,result)


if __name__=="__main__":
    main()
