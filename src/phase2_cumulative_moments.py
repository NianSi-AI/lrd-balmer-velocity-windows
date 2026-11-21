"""Phase 2. Run 'inject' then 'measure', with a new output directory containing protocol.json."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
C_KMS = 299792.458
HA_REST = 6564.614
SQRT_8LN2 = np.sqrt(8*np.log(2))
NII = (np.array([6549.86, 6585.27])/HA_REST-1)*C_KMS/1000
OI = (6365.54/HA_REST-1)*C_KMS/1000


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inputs():
    rows = read_csv(ROOT / "data/halpha_sources.csv")
    assert len(rows) == 225
    return list(enumerate(rows))


def acquire(obj):
    index, row = obj
    with np.load(ROOT / "data/halpha_pixels.npz", allow_pickle=False) as archive:
        v, y, e, good = archive[f"s{index:03d}"].T
    meta = {**row, **{key: float(row[key]) for key in ("h", "snr", "sg", "area", "ls")}}
    meta["abs_flag"] = row["abs_flag"] == "True"
    return meta, v.copy(), y.copy(), e.copy(), good.astype(bool)


def geometry(v,good,windows):
    mid = (v[:-1]+v[1:])/2
    edge = np.r_[2*v[0]-mid[0],mid,2*v[-1]-mid[-1]]
    # Do not assign missing wavelength intervals to their nearest good sample.
    half = .55*np.median(np.diff(v))
    left = np.maximum(edge[:-1],v-half)[good]
    right = np.minimum(edge[1:],v+half)[good]
    mats=[]
    for width in windows:
        lo,hi = np.maximum(left,-width),np.minimum(right,width)
        hi=np.maximum(lo,hi)
        mats.append([hi-lo,(hi**2-lo**2)/2,(hi**3-lo**3)/3])
    return np.asarray(mats)


def moments(mat, y, covered):
    flux,first,second = (mat@y).T
    with np.errstate(invalid="ignore", divide="ignore"):
        var=second/flux-(first/flux)**2
        return np.where(covered & (flux>0) & (var>0),1000*np.sqrt(np.maximum(var,0)),np.nan)


def gauss(x,mu,s):
    return np.exp(-.5*((x-mu)/s)**2)


def narrow_and_broad(x,p,ls,jacobian=False):
    an,ai,cn,sn,a1,a2,cb,s1,s2 = p
    s = np.hypot(sn,ls)
    offsets = np.r_[cn,cn+NII]
    diff = x[:,None]-offsets
    g = np.exp(-.5*(diff/s)**2)
    amp = np.array([an,ai/2.98,ai])
    narrow = g@amp
    dg1, dg2 = x-cb, x-cb
    b1,b2 = gauss(x,cb,s1),gauss(x,cb,s2)
    if not jacobian:
        return narrow,narrow+a1*b1+a2*b2
    return np.column_stack((g[:,0],g[:,1]/2.98+g[:,2],(g*diff)@amp/s**2,
             (g*diff**2)@amp*sn/s**4,b1,b2,a1*b1*dg1/s1**2+a2*b2*dg2/s2**2,
             a1*b1*dg1**2/s1**3,a2*b2*dg2**2/s2**3))


class Extraction:
    def __init__(self,meta,raw,err,good,windows):
        self.meta=meta
        self.mat=geometry(raw,good,windows)
        self.coverage=self.mat[:,0,:].sum(axis=1)/(2*windows)
        self.v,self.e=raw[good],err[good]
        v=self.v
        self.side=(((v>=-11)&(v<=-6.5))|((v>=8.7)&(v<=11))) & (abs(v-OI)>.45)
        self.has_side=all(np.sum(self.side & (v*sign>0))>=4 for sign in [-1,1])
        self.design=np.column_stack([np.ones(len(v)),v])
        self.inv=np.linalg.pinv(self.design[self.side]/self.e[self.side,None])
        self.inner=abs(v)<=4.5
        self.ready=self.has_side and self.inner.sum()>=18

    def continuum(self,y):
        return self.design@(self.inv@(y[self.side]/self.e[self.side]))

    def extract(self,y):
        if not self.ready:
            return np.full((2,len(self.mat)),np.nan),False
        clean=y-self.continuum(y)
        x,e=self.v[self.inner],self.e[self.inner]
        target=clean[self.inner]
        peak=max(np.percentile(target,95),1.)
        init=[peak*.25,peak*.12,0,.12,peak*.7,peak*.25,0,.7,2.]
        low=[0,0,-.55,.025,0,0,-.65,.21,.85]
        high=[np.inf,np.inf,.55,.26,np.inf,np.inf,.65,2.3,6.]
        ls=self.meta["ls"]
        fit=least_squares(lambda p:(narrow_and_broad(x,p,ls)[1]-target)/e,init,
                          jac=lambda p:narrow_and_broad(x,p,ls,True)/e[:,None],bounds=(low,high),
                          loss="soft_l1",x_scale="jac",max_nfev=240,ftol=1e-6,xtol=1e-6)
        narrow=narrow_and_broad(self.v,fit.x,ls)[0]
        total=moments(self.mat,clean,self.coverage>=.98)
        broad=moments(self.mat,clean-narrow,self.coverage>=.98) if fit.success else np.full(len(total),np.nan)
        return np.stack([broad,total]),bool(fit.success)

    def noise_bank(self,real):
        s=self.side
        r=(real-self.continuum(real))/self.e
        r=r-np.median(r[s])
        dispersion=np.sqrt(np.mean(r[s]**2))
        r=r/max(min(dispersion,1.),1e-6)
        ids=np.flatnonzero(s)
        breaks=(np.diff(ids)!=1)|(np.diff(self.v[ids])>1.6*np.median(np.diff(self.v)))
        chunks=np.split(ids,np.flatnonzero(breaks)+1)
        return np.concatenate([np.lib.stride_tricks.sliding_window_view(r[g],4) for g in chunks if len(g)>=4])

    def simulate(self,rng,real,bank,kind,draw,core_fwhm=None):
        v, m = self.v,self.meta
        # All velocities here in 1000 km/s; area is in the normalized flux-density units times 1000 km/s.
        fwhm=(2.1 if m["survey"]=="JWST" else 3.5)*10**(.18*(m["h"]-(42.5 if m["survey"]=="JWST" else 43.))+.05*draw[0])
        s=(fwhm if core_fwhm is None else core_fwhm)/SQRT_8LN2
        fraction=0 if kind=="Gaussian" else .05+.30*draw[1]
        def obs_profile(x):
            density=np.zeros_like(x)
            for weight,width in [(1-fraction,s),(fraction,3*s)]:
                mu=.09*draw[2]
                obs_s=np.hypot(width,m["ls"])
                density+=weight*gauss(x,mu,obs_s)/obs_s
                if draw[3]<.4:
                    # Attenuate broad light before the instrumental convolution (Gaussian product).
                    abw=.14
                    abmu=-.27
                    var=1/(1/width**2+1/abw**2)
                    center=var*(mu/width**2+abmu/abw**2)
                    pref=.5*np.exp(-.5*(mu-abmu)**2/(width**2+abw**2))*np.sqrt(var)/width
                    out_s=np.sqrt(var+m["ls"]**2)
                    density-=weight*pref*gauss(x,center,out_s)/out_s
            return m["area"]*density/np.sqrt(2*np.pi)
        broad=obs_profile(v)
        idx=rng.integers(len(bank),size=(len(v)+3)//4)
        noise=(bank[idx]*rng.choice([-1,1],size=(len(idx),1))).ravel()[:len(v)]*self.e
        ampl=m["area"]/(s*np.sqrt(2*np.pi))
        narrow=narrow_and_broad(v,[ampl*draw[4],ampl*.35*draw[5],.12*draw[6],.1,0,0,0,.7,2],m["ls"])[0]
        return self.continuum(real)+broad+narrow+noise,obs_profile


def process(task):
    phase, obj, limits, seed, repetitions = task
    m,v,y,e,good=acquire(obj)
    windows=np.array(limits)/1000
    ext=Extraction(m,v,e,good,windows)
    real=y[good]
    rng=np.random.default_rng(seed)
    base={key:m[key] for key in ["survey","source","h","snr","abs_flag","sg"]}
    rows=[]
    if phase=="measure":
        (broad,total),fit_ok=ext.extract(real)
        for j,w in enumerate(limits):
            rows.append(dict(**base,V=w,coverage=ext.coverage[j],sidebands_ok=ext.has_side,fit_ok=fit_ok,
                             sigma=broad[j],sigma_total=total[j]))
        return rows
    dense=np.arange(-6.,6.0001,.005)
    fine_mat=geometry(dense,np.ones(len(dense),bool),windows)
    if ext.ready:
        bank=ext.noise_bank(real)
    for rep in range(repetitions):
        draw=[rng.normal(),rng.random(),rng.normal(),rng.random(),rng.uniform(.1,1.5),rng.uniform(.05,1),rng.normal()]
        for kind in ("Gaussian","variable wings"):
            if ext.ready:
                synthetic,function=ext.simulate(rng,real,bank,kind,draw)
                (recovered,_),_=ext.extract(synthetic)
                truth=moments(fine_mat,function(dense),np.ones(len(windows),bool))
            else:
                recovered=truth=np.full(len(windows),np.nan)
            for j,w in enumerate(limits):
                rows.append(dict(**base,V=w,profile=kind,rep=rep,truth=truth[j],recovered=recovered[j]))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["inject","measure"])
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    proto=args.directory/"protocol.json"
    cfg=json.loads(proto.read_text())
    marker=args.directory/"injection_complete.sha256"
    digest=hashlib.sha256(proto.read_bytes()).hexdigest()
    if args.phase=="measure":
        assert marker.read_text().strip()==digest
    grid=np.arange(-10.,10.001,.01)
    mat=geometry(grid,np.ones(len(grid),bool),np.array([2.,9.]))
    np.testing.assert_allclose(moments(mat,np.ones(len(grid)),True),1000*np.array([2.,9.])/np.sqrt(3),rtol=1e-8)
    np.testing.assert_allclose(moments(mat,gauss(grid,0,1),True)[-1],1000,rtol=1e-5)
    tasks=[(args.phase,obj,cfg["windows_kms"],cfg["seed"]+i,10) for i,obj in enumerate(inputs())]
    out=args.directory/("injections.csv" if args.phase=="inject" else "measurements.csv")
    if out.exists():
        raise FileExistsError(out)
    with ProcessPoolExecutor(max_workers=5) as pool:
        records=[]
        for i,result in enumerate(pool.map(process,tasks),1):
            records.extend(result)
            if i%50==0: print(args.phase,i,flush=True)
    write_csv(out,records)
    if args.phase=="inject":
        with marker.open("x") as stream: print(digest,file=stream)
    print(out,len(records))


if __name__=="__main__":
    main()
