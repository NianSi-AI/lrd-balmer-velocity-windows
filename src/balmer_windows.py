"""Integrate native Balmer pixels and recover the same-grating template contrast."""
from pathlib import Path
import csv
import json
import numpy as np
from phase2_cumulative_moments import write_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "build/tables"
C_KMS, HA_REST, HB_REST = 299792.458, 6564.614, 4862.683
N, BLOCK = 8192, 4

def footprint(full_grid, selected, side, low, high):
    # Define edges before applying spectral masks to avoid bridging bad pixels.
    assert np.all(np.diff(full_grid) > 0)
    mid = (full_grid[:-1] + full_grid[1:]) / 2
    edges = np.r_[2 * full_grid[0] - mid[0], mid, 2 * full_grid[-1] - mid[-1]]
    index = np.searchsorted(full_grid, selected)
    np.testing.assert_allclose(full_grid[index], selected)
    intervals = []
    if side in ("BOTH", "BLUE"):
        intervals.append((-high, -low))
    if side in ("BOTH", "RED"):
        intervals.append((low, high))
    return sum(np.maximum(0, np.minimum(edges[index+1], b) - np.maximum(edges[index], a))
               for a, b in intervals)


BINS = [(0, 300), (300, 1000), (1000, 1500), (1500, 2500), (2500, 3500)]
HB_SIDEBANDS = [(4705, 4770), (5040, 5140), (5240, 5320)]


def line_fluxes(wave, flux, error, meta, rest, alpha):
    lam = rest * (1 + meta["z_OIII"])
    v = C_KMS * (wave / lam - 1)
    good = np.isfinite(flux + error) & (error > 0)
    if alpha:
        anchors = ((v > -9000) & (v < -5200)) | ((v > 5200) & (v < 6400)) | ((v > 8300) & (v < 9500))
    else:
        wr = wave / (1 + meta["z_OIII"])
        anchors = np.logical_or.reduce([(wr >= lo) & (wr <= hi) for lo, hi in HB_SIDEBANDS])
    side = good & anchors
    region = good & (abs(v) < 5100)
    assert np.any(side & (v < 0)) and np.any(side & (v > 0))
    y, e = (a * lam / C_KMS / 1e-17 for a in (flux[good], error[good]))
    x = np.column_stack((np.ones(len(y)), v[good] / 5000))
    mask = side[good]
    # Map sideband spectra to the fitted straight-line continuum.
    coef = np.linalg.pinv(x[mask] / e[mask,None]) / e[mask]
    vv = v[region]
    kernel = np.eye(len(vv))
    if alpha:
        sigma = meta["delta_sigma_match_kms"]
        kernel = np.exp(-0.5*((vv[:,None]-vv[None,:])/sigma)**2) * np.gradient(vv)[None,:]
        kernel /= kernel.sum(axis=1, keepdims=True)
    rows = []
    for low, high in BINS:
        weights = footprint(v, vv, "BOTH", low, high)
        np.testing.assert_allclose(weights.sum(), 2*(high-low))
        weights = weights @ kernel
        row = np.zeros(len(y))
        row[region[good]] = weights
        row[mask] -= (weights @ x[region[good]]) @ coef
        # Line-free synthetic constant and sloped spectra integrate to zero.
        np.testing.assert_allclose(row @ x, 0, atol=1e-8)
        rows.append(row)
    operator = np.array(rows)
    noise = operator * e
    return operator @ y, noise @ noise.T, int(mask.sum())


def measure_windows():
    metadata = json.loads((DATA / "balmer_sources.json").read_text())
    table_a, table_b = [], []
    names = ["center_lt300", "core_norm_300_1000", "inner_1000_1500",
             "shoulder_1500_2500", "wing_2500_3500"]
    with np.load(DATA / "balmer_pixels.npz", allow_pickle=False) as archive:
        for name, meta in metadata.items():
            def field(key):
                return archive[f"{name}_{key}"]
            wave, flux, error = (field(k) for k in ("wave", "flux", "error"))
            ga, gb = [C_KMS * (wave / (rest * (1 + meta["z_OIII"])) - 1)
                      for rest in (HA_REST, HB_REST)]
            va, vb, ya, yb, cov, eb, unit = (field(k) for k in ("va","vb","ya","yb","cov","eb","unit"))
            a, ca, na = line_fluxes(wave, flux, error, meta, HA_REST, True)
            b, cb, nb = line_fluxes(wave, flux, error, meta, HB_REST, False)
            for i, (low, high) in enumerate(BINS):
                for side in (("BOTH","BLUE","RED") if i == 1 else ("BOTH",)):
                    wa = footprint(ga, va, side, low, high)
                    wb = footprint(gb, vb, side, low, high)
                    expected = (high-low) * (2 if side == "BOTH" else 1)
                    np.testing.assert_allclose([wa.sum(), wb.sum()], expected, atol=0.01)
                    fa, fb = (wa @ ya)*unit, (wb @ yb)*unit
                    ea, e_b = np.sqrt(wa @ cov @ wa)*unit, np.linalg.norm(wb*eb)*unit
                    table_a.append(dict(source=name,window=names[i],side=side,F_Ha_1e17=fa,
                        e_Ha_pixel_1e17=ea,F_Hb_1e17=fb,e_Hb_pixel_1e17=e_b,
                        D_point_if_Hb_ge_2_pixel_sigma=fa/fb if fb>=2*e_b and fa>0 else ""))
                ea, e_b = np.sqrt(ca[i,i]), np.sqrt(cb[i,i])
                table_b.append(dict(source=name,v_min_kms=low,v_max_kms=high,
                    total_F_Ha_1e17=a[i],e_Ha_1e17=ea,total_F_Hb_1e17=b[i],e_Hb_1e17=e_b,
                    total_D_if_Hb_ge_2sigma=a[i]/b[i] if b[i]>=2*e_b and a[i]>0 else ""))
    write_csv(OUT / "three_sources_observed_native_windows.csv", table_a)
    write_csv(OUT / "total_unsubtracted_balmer.csv", table_b)


def template_groups():
    with (DATA / "template_sources.csv").open() as f:
        sources = list(csv.DictReader(f))
    seeds = np.random.SeedSequence(20261121).spawn(len(sources))
    point, noise, bright = [], [], []
    with np.load(DATA / "template_pixels.npz",allow_pickle=False) as pixels:
        for r, seed in zip(sources,seeds,strict=True):
            prefix = r["source"] + "_"
            coeff, signals, bank, lengths, core, observed = (pixels[prefix+k] for k in
                ("coeff","signals","bank","lengths","core","observed"))
            np.testing.assert_allclose(coeff @ signals, np.eye(2), atol=1e-7)
            assert sum(lengths) == coeff.shape[1]
            rng = np.random.default_rng(seed)
            errors = np.zeros((N,2))
            start = 0
            for length in lengths:
                count = (length+BLOCK-1)//BLOCK
                idx = rng.integers(len(bank),size=(N,count))
                signs = rng.choice([-1,1],size=(N,count,1))
                trial = (bank[idx]*signs).reshape(N,-1)[:,:length]
                errors += trial @ coeff[:,start:start+length].T
                start += length
            point.append([float(core), coeff[1] @ observed])
            noise.append(errors)
            bright.append(bool(int(r["top_three"])))
    full = np.asarray(noise)
    weights = 1 / np.var(full[:,:N//2,0]-full[:,:N//2,1],axis=1,ddof=1)
    weights /= weights.sum()
    noise = full[:,N//2:,:]
    bright = np.array(bright)
    result = {}
    for label, keep in (("all33", np.ones(len(bright),bool)), ("top3",bright), ("other30",~bright)):
        w = weights[keep]
        norm = w/w.sum()
        core, outer = norm @ np.asarray(point)[keep]
        delta = np.einsum("s,sij->ij",norm,noise[keep])
        assert np.all(core-delta[:,0]>0)
        ci = np.quantile((outer-delta[:,1])/(core-delta[:,0]),[.025,.975])
        result[label] = dict(n=int(keep.sum()),archived_weight=float(w.sum()),
            wing_over_core=outer/core,ci95_pixel_conditional=ci.tolist())
    (OUT / "template_groups.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    measure_windows()
    template_groups()
