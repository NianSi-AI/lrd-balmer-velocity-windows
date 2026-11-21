"""Full-sample comparison with the pre-announced supplemental edge sideband."""
import argparse
from pathlib import Path
import numpy as np
from phase2_moment_results import load, predictive, select, scatter, write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    out=parser.parse_args().directory
    measured=load(out/"edge_sensitivity_measurements.csv")
    sim=load(out/"source_conditioned_injections.csv")
    rng=np.random.default_rng(221122)
    predicted=[r for g in ["JWST","SDSS"] for r in predictive(measured,sim,g,rng)]
    predfile=out/"edge_conditioned_predictions.csv"
    if predfile.exists():
        saved=load(predfile)
        np.testing.assert_allclose(saved["predicted_delta_med"],[r["predicted_delta_med"] for r in predicted])
    else:
        write_csv(predfile,predicted)
    results=[]
    for w in [2500,3500,4500]:
        point=[]; draws=[]
        for name in ["JWST","SDSS"]:
            t=select(measured,name)
            t=t[(t["V"]==w)&np.isfinite(t["sigma"])]
            h=t["h"]
            y=np.log10(t["sigma"])
            i=rng.integers(len(h),size=(10000,len(h)))
            point.append(scatter(h,y))
            draws.append(scatter(h[i],y[i]))
        lo,hi=np.quantile(draws[0]-draws[1],[.025,.975])
        results.append(dict(V=w,JWST_minus_SDSS=point[0]-point[1],lo95=lo,hi95=hi))
    write_csv(out/"cross_survey_scatter_difference.csv",results)


if __name__ == "__main__":
    main()
