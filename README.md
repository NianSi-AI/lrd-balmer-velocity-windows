# Balmer velocity windows in little red dots

Reproduction package for manuscript v3.1. This repository measures how the observed residual scatter in broad Hα line dispersion at fixed Hα luminosity changes with the velocity window. It also reconstructs the velocity-binned Hβ/Hα ratios of three JWST sources and compares integrated widths in JWST, SDSS and Euclid.

## Run

Tested with Python 3.12.4. From the repository root, in a fresh checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --pdf
```

The PDF option requires a TeX installation with latexmk, pdfLaTeX, BibTeX, the AASTeX 6.3.1 class and the aasjournal bibliography style. Use `python reproduce.py` for the full numerical and graphical analysis without TeX. Once dependencies are installed, the build runs offline. Each run writes to an empty `build/` directory. To retain and compare successive runs, move the previous build aside.

The main outputs are `build/main.pdf` (with the PDF option), five PDF/PNG figure pairs in `build/figures/`, machine-readable results in `build/tables/`, and the values used in the four manuscript tables and text in `build/numbers.tex`. The entry point checks every numerical macro used in v3.1 at its displayed precision before drawing the figures. Font availability can change rendering across systems.

## What is recomputed

- **Figure 1, Figure A1 and the window-scatter and calibration tables.** Start with the extracted Hα pixels of 40 continuum-clean JWST sources and 185 accepted SDSS sources. The code injects both broad-profile families on each native grid, refits the continuum and narrow lines, integrates signed finite-window moments, and runs source-level paired bootstraps and source-conditioned simulations. Default sidebands retain 20 high-S/N JWST sources and 167 paired SDSS sources at 2500 and 4500 km/s. The JWST scatter grows from 0.0358 to 0.0837 dex. The supplementary sidebands recover all 22 high-S/N JWST sources.
- **Figure 2 and the velocity-binned Balmer table.** Reintegrate three sources on exact native pixel footprints in the [O III] systemic frame. Treatment A starts with the archived best-fit continuum/narrow-line subtraction and matched Hα residual covariance. Treatment B independently fits the continuum to the extracted spectra. Errors and detection flags are recomputed in each window.
- **Figure 3 and the sample table.** Refit the single-Gaussian FWHM–luminosity relation to each final survey catalog separately. Each JWST luminosity is the integrated broad flux of the BIC-selected line profile, held fixed across velocity windows. The shared luminosity interval uses the final Planck18 bridge catalog.
- **Figure B1.** Use the native-pixel template operators and empirical residual banks of 33 same-grating sources. Regenerate the 8192 block-noise realizations per source, calibrate the weights on the first half and measure conditional intervals on the held-out half.

The starting products are extracted spectra, fitted nuisance terms, and final catalog quantities. The pipeline retains the archival source classification, Hα velocity centers, integrated luminosities, line-spread estimates and Euclid selection/posterior summary. Detector reduction, upstream catalog fits and the original Euclid blind inference precede this release. Observational data sources and method references appear in Sections 2–3 and the bibliography of `manuscript/main.tex`.

## Inputs

All NPZ files contain plain numeric NumPy arrays, readable with `allow_pickle=False`. Archive member names and row order are part of the data format.

| Input in `data/` | Contents |
|---|---|
| `halpha_sources.csv`, `halpha_pixels.npz` | Source index in CSV order, matching keys `s000` through `s224`. Each array has four columns: velocity relative to the fitted Hα center in 1000 km/s, normalized flux density, normalized 1σ error, and good-pixel flag. Flux and error share the median good-pixel error scale. `h` is log10 broad Hα luminosity in erg/s, `snr` broad-line S/N, `sg` single-Gaussian FWHM in km/s, `ls` instrumental dispersion in 1000 km/s, and `area` integrated broad flux in the normalized flux-density and velocity units. Preserve invalid pixels and masks. |
| `protocol.json`, `sensitivity_protocol.json` | Native-window integration, noise simulation, recovery thresholds, and supplementary continuum-sideband specification. |
| `balmer_sources.json`, `balmer_pixels.npz` | Per source, systemic redshift, grating, integrated FWHM, matching kernel width and combined-exposure count. NPZ keys have the source name as prefix. `wave,flux,error` are observed wavelength in Å and observed flux density and error in CGS per Å. `va,vb` are Hα/Hβ native velocities in km/s. `ya,yb` are fitted-nuisance-subtracted residuals, `cov` the matched Hα pixel covariance, `eb` the Hβ diagonal standard errors. `unit` converts the velocity integrals to flux in units of 10⁻¹⁷ erg s⁻¹ cm⁻². |
| `template_sources.csv`, `template_pixels.npz` | Source order and three-source subset. Per-source arrays contain the core/wing linear operators (`coeff`), two unit templates (`signals`), four-pixel noise blocks (`bank`), contiguous segment lengths, measured core amplitude and error-normalized observed pixels. |
| `jwst.csv`, `sdss.csv`, `euclid.csv` | Published-sample catalog columns used for sample descriptors and integrated widths. JWST source keys map to canonical target names. SDSS uses its coordinate identifier. Euclid `w` is log10 FWHM in km/s. All three catalogs use `h` for log10 observed Hα luminosity in erg/s. |
| `bridge.csv`, `survey_summary.json` | Source IDs and luminosities for the common Hα interval, plus the archived Euclid selection limit, parent size, conditional scatter posterior quantiles (2.5, 16, 50, 84, 97.5 percent), and SDSS attempted-spectra count. |

## Numerical outputs

`observed_scatter.csv` reports counts, scatter, paired increments and bootstrap intervals. `calibration_summary.csv` reports usable fractions, log-width biases and recovery errors. `source_conditioned_predictions.csv` holds the conditional predictions for the default-sideband sample. The files beginning with `edge_` describe the supplementary sidebands. `three_sources_observed_native_windows.csv` and `total_unsubtracted_balmer.csv` give the two Balmer treatments. `template_groups.json` gives all-source, top-three and remaining-source contrasts. The science figures and manuscript read these regenerated products in the same build.

