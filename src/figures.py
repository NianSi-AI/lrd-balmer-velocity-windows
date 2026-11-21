import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "build/tables"
OUT_DIR = ROOT / "build/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_apj_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 9.8,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.85,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
    })


def save_fig(fig, stem):
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=250)
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")


def make_fig1():
    """Main Figure 1: Unified default-sideband s(V) scan (a) and paired increments + 20-source predictions (b)."""
    with (TABLES / "observed_scatter.csv").open() as f:
        obs = {(r["group"], int(r["V"])): r for r in csv.DictReader(f)}
    with (TABLES / "source_conditioned_predictions.csv").open() as f:
        pred_preset = {(r["group"], r["profile"]): r for r in csv.DictReader(f)}

    c_jwst = "#9E1F3A"   # JWST default sidebands (bold crimson)
    c_sdss = "#1F5E8C"   # SDSS default sidebands (bold steel blue)
    c_pred = "#6B6661"   # Conditional prediction interval slate

    fig = plt.figure(figsize=(10.2, 3.85))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.08, 1.12], wspace=0.36)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    vs_4 = [1500, 2500, 3500, 4500]
    vk_4 = np.array(vs_4, dtype=float) / 1000.0

    # Dynamic sample counts from observed_scatter.csv
    n_jwst = int(obs[("JWST", 2500)]["n"])
    sd_ns = [int(obs[("SDSS", V)]["n"]) for V in vs_4]
    n_sd_min, n_sd_max = min(sd_ns), max(sd_ns)
    n_sd_paired = int(obs[("SDSS", 4500)]["paired_n"])
    n_mock_g = int(pred_preset[("JWST", "Gaussian")]["n"])
    n_mock_w = int(pred_preset[("JWST", "variable wings")]["n"])
    assert n_jwst == n_mock_g == n_mock_w

    # --- Panel (a): Validated observed scatter curves s(V), default sidebands ---
    jw_s = np.array([float(obs[("JWST", V)]["s"]) for V in vs_4])
    jw_lo = np.array([float(obs[("JWST", V)]["lo"]) for V in vs_4])
    jw_hi = np.array([float(obs[("JWST", V)]["hi"]) for V in vs_4])

    sd_s = np.array([float(obs[("SDSS", V)]["s"]) for V in vs_4])
    sd_lo = np.array([float(obs[("SDSS", V)]["lo"]) for V in vs_4])
    sd_hi = np.array([float(obs[("SDSS", V)]["hi"]) for V in vs_4])

    ax0.fill_between(vk_4, jw_lo, jw_hi, color=c_jwst, alpha=0.15, lw=0, zorder=2)
    ax0.plot(
        vk_4, jw_s, color=c_jwst, ls="-", lw=2.2, marker="o", ms=5.7,
        label=f"JWST ({n_jwst}, default sidebands)", zorder=5
    )

    ax0.fill_between(vk_4, sd_lo, sd_hi, color=c_sdss, alpha=0.14, lw=0, zorder=1)
    ax0.plot(
        vk_4, sd_s, color=c_sdss, ls="--", lw=2.0, marker="s", ms=5.2,
        label=f"SDSS ({n_sd_min}–{n_sd_max}, default sidebands)", zorder=4
    )

    ax0.set_xlim(1.25, 4.75)
    ax0.set_ylim(0.015, 0.112)
    ax0.set_xticks(vk_4)
    ax0.set_xticklabels(["1.5", "2.5", "3.5", "4.5"])
    ax0.set_xlabel(r"Velocity-window half-width $V\ (10^3\ \mathrm{km\,s^{-1}})$")
    ax0.set_ylabel(r"Observed residual scatter $s(V)$ at fixed $L_{\mathrm{H}\alpha}$ (dex)")
    ax0.set_title(r"(a) Observed scatter vs. window half-width", pad=7)
    ax0.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", borderpad=0.36)
    ax0.grid(axis="y", color="#EAEAEA", lw=0.6)

    # --- Panel (b): Paired increments V = 2500 -> 4500 km/s (unified default sidebands) ---
    ax1.axvline(0.0, color="#777777", ls="--", lw=0.9, zorder=1)
    ax1.axhline(2.15, color="#D8D8D8", lw=0.9, ls="-", zorder=1)

    jw_delta = float(obs[("JWST", 4500)]["delta"])
    jw_dlo = float(obs[("JWST", 4500)]["delta_lo"])
    jw_dhi = float(obs[("JWST", 4500)]["delta_hi"])

    sd_delta = float(obs[("SDSS", 4500)]["delta"])
    sd_dlo = float(obs[("SDSS", 4500)]["delta_lo"])
    sd_dhi = float(obs[("SDSS", 4500)]["delta_hi"])

    # Subtle vertical reference line from JWST (20) delta through the 20-source prediction block
    ax1.plot([jw_delta, jw_delta], [-0.15, 3.65], color=c_jwst, ls=":", lw=1.0, alpha=0.55, zorder=2)

    # Subtle group labels inside Panel (b)
    ax1.text(0.005, 4.22, "Observed paired increments", fontsize=7.8,
             color="#333333", ha="left", va="top", style="italic")
    ax1.text(0.005, 1.98, f"Same {n_jwst} JWST sources: mock predictions", fontsize=7.8,
             color="#444444", ha="left", va="top", style="italic")

    data_rows = [
        (3.65, f"JWST ({n_jwst})", jw_delta, jw_dlo, jw_dhi, c_jwst, "o"),
        (2.75, f"SDSS ({n_sd_paired})", sd_delta, sd_dlo, sd_dhi, c_sdss, "s"),
    ]
    for y, _, mid, lo, hi, col, mk in data_rows:
        ax1.errorbar(
            mid, y, xerr=[[mid - lo], [hi - mid]],
            fmt=mk, ms=5.8, color=col, mfc=col, mec="white", mew=0.5,
            elinewidth=2.0, capsize=3.3, capthick=1.2, zorder=4
        )

    pred_rows = [
        (1.30, f"Gaussian ({n_mock_g})",
         float(pred_preset[("JWST", "Gaussian")]["predicted_delta_med"]),
         float(pred_preset[("JWST", "Gaussian")]["predicted_delta_lo"]),
         float(pred_preset[("JWST", "Gaussian")]["predicted_delta_hi"]),
         "o"),
        (0.45, f"Core + wing ({n_mock_w})",
         float(pred_preset[("JWST", "variable wings")]["predicted_delta_med"]),
         float(pred_preset[("JWST", "variable wings")]["predicted_delta_lo"]),
         float(pred_preset[("JWST", "variable wings")]["predicted_delta_hi"]),
         "^"),
    ]
    band_h = 0.36
    for y, _, med, lo, hi, mk in pred_rows:
        rect = mpatches.Rectangle(
            (lo, y - band_h / 2.0), hi - lo, band_h,
            facecolor="#EFECE6", edgecolor=c_pred, lw=1.05, ls="--", zorder=2
        )
        ax1.add_patch(rect)
        ax1.plot(med, y, marker=mk, ms=5.2, mfc="white", mec=c_pred, mew=1.25, zorder=3)

    all_y = [r[0] for r in data_rows] + [r[0] for r in pred_rows]
    all_lbl = [r[1] for r in data_rows] + [r[1] for r in pred_rows]
    ax1.set_yticks(all_y)
    ax1.set_yticklabels(all_lbl, fontsize=8.5)
    ax1.set_xlim(-0.012, 0.236)
    ax1.set_ylim(-0.15, 4.42)
    ax1.set_xlabel(r"Paired scatter increment $\Delta s_{2500\to 4500}$ (dex)")
    ax1.set_title(r"(b) Paired increment ($2500 \to 4500\ \mathrm{km\,s^{-1}}$)", pad=7)
    ax1.grid(axis="x", color="#EAEAEA", lw=0.6)
    ax1.tick_params(axis="y", length=0, pad=5)

    h_boot = ax1.errorbar([], [], xerr=[[0.01], [0.01]], fmt="o", ms=5.0, color="#333333",
                          elinewidth=1.8, capsize=3.0, label="95% source-bootstrap CI")
    h_pred = mpatches.Patch(facecolor="#EFECE6", edgecolor=c_pred, lw=1.0, ls="--",
                            label="95% cond. prediction interval")
    ax1.legend(handles=[h_boot, h_pred], loc="upper right", bbox_to_anchor=(0.985, 0.90),
               frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=7.6, borderpad=0.32)

    save_fig(fig, "fig1_cumulative_dispersion")


def make_fig2():
    """Main Figure 2: Single-row 3-panel Balmer ratio R = F(Hb)/F(Ha), ordered 55604 -> 42046 -> 68797."""
    with (TABLES / "three_sources_observed_native_windows.csv").open() as f:
        schA = {(r["source"], r["window"], r["side"]): r for r in csv.DictReader(f)}
    with (TABLES / "total_unsubtracted_balmer.csv").open() as f:
        schB = {(r["source"], int(r["v_min_kms"]), int(r["v_max_kms"])): r for r in csv.DictReader(f)}
    meta = json.loads((DATA / "balmer_sources.json").read_text())

    c_schA = "#1D5B8A"
    c_schB = "#BA4A22"

    fig = plt.figure(figsize=(10.6, 3.65))
    gs = gridspec.GridSpec(1, 3, wspace=0.07)

    sources = [
        ("RUBIES-EGS-55604", "(a) RUBIES-EGS-55604"),
        ("RUBIES-EGS-42046", "(b) RUBIES-EGS-42046"),
        ("JADES-GN-68797", "(c) JADES-GN-68797"),
    ]
    win_specs = [
        ("center_lt300", 0, 300, "0–0.3"),
        ("core_norm_300_1000", 300, 1000, "0.3–1"),
        ("inner_1000_1500", 1000, 1500, "1–1.5"),
        ("shoulder_1500_2500", 1500, 2500, "1.5–2.5"),
        ("wing_2500_3500", 2500, 3500, "2.5–3.5"),
    ]
    x_idx = np.arange(len(win_specs), dtype=float)

    axes = []
    for idx, (sname, title) in enumerate(sources):
        ax = fig.add_subplot(gs[idx])
        axes.append(ax)
        ax.axhline(0.0, color="#888888", lw=0.85, ls="-", zorder=1)
        ax.grid(axis="y", color="#EAEAEA", lw=0.6, zorder=0)

        for j, (wname, vmin, vmax, _) in enumerate(win_specs):
            ra = schA[(sname, wname, "BOTH")]
            fa, efa = float(ra["F_Ha_1e17"]), float(ra["e_Ha_pixel_1e17"])
            fb, efb = float(ra["F_Hb_1e17"]), float(ra["e_Hb_pixel_1e17"])
            rA = fb / fa
            erA = math.sqrt((efb / fa) ** 2 + (fb * efa / fa ** 2) ** 2)
            detA = ra["D_point_if_Hb_ge_2_pixel_sigma"] != ""
            ax.errorbar(
                x_idx[j] - 0.13, rA, yerr=erA, fmt="o", ms=5.2,
                color=c_schA, mfc=c_schA if detA else "white", mew=1.2,
                elinewidth=1.15, capsize=2.4, zorder=4,
                label="Narrow lines removed (A)" if (idx == 0 and j == 0) else None
            )

            rb = schB[(sname, vmin, vmax)]
            fa_b, efa_b = float(rb["total_F_Ha_1e17"]), float(rb["e_Ha_1e17"])
            fb_b, efb_b = float(rb["total_F_Hb_1e17"]), float(rb["e_Hb_1e17"])
            rB = fb_b / fa_b
            erB = math.sqrt((efb_b / fa_b) ** 2 + (fb_b * efa_b / fa_b ** 2) ** 2)
            detB = rb["total_D_if_Hb_ge_2sigma"] != ""
            ax.errorbar(
                x_idx[j] + 0.13, rB, yerr=erB, fmt="s", ms=4.8,
                color=c_schB, mfc=c_schB if detB else "white", mew=1.2,
                elinewidth=1.15, capsize=2.4, zorder=4,
                label="Continuum only (B)" if (idx == 0 and j == 0) else None
            )

        z_oiii = float(meta[sname]["z_OIII"])
        grat = meta[sname]["grating"]
        ax.set_title(rf"{title}" + "\n" + rf"($z_{{[\mathrm{{O\,III}}]}}={z_oiii:.2f}$, {grat})", fontsize=9.3, pad=5)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([w[3] for w in win_specs], rotation=22, ha="right", fontsize=8.4)
        ax.set_xlim(-0.55, len(win_specs) - 0.45)
        ax.set_ylim(-0.115, 0.236)
        if idx == 0:
            ax.set_ylabel(r"Velocity-binned flux ratio $R = F(\mathrm{H}\beta)/F(\mathrm{H}\alpha)$")
        else:
            ax.tick_params(axis="y", labelleft=False)

    axes[1].set_xlabel(r"Symmetric Doppler-velocity interval $|v|$ relative to $[\mathrm{O\,III}]$ ($10^3\ \mathrm{km\,s^{-1}}$)", labelpad=5)
    handles, labels = axes[0].get_legend_handles_labels()
    open_proxy = mpl.lines.Line2D([0], [0], marker="o", ms=5.0, mfc="white", mec="#444444", ls="none", mew=1.15)
    handles.append(open_proxy)
    labels.append(r"Open: $\mathrm{H}\beta$ below $2\sigma$")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.095),
               ncol=3, frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=8.2, columnspacing=1.6)

    save_fig(fig, "fig2_balmer_velocity_windows")


def _fit_ols_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sd = float(np.std(resid, ddof=1))
    return float(slope), float(intercept), sd


def make_fig3():
    """Main Figure 3: Three-survey integrated single-Gaussian FWHM vs observed L_Ha on common axes."""
    with (DATA / "jwst.csv").open() as f:
        jw40 = [r for r in csv.DictReader(f) if r["cont_qc_pass"] == "True"]
    with (TABLES / "measurements.csv").open() as f:
        jw20_ids = set(
            r["source"] for r in csv.DictReader(f)
            if r["survey"] == "JWST" and float(r["snr"]) >= 15.0 and int(float(r["V"])) == 2500 and r["fit_ok"] == "True"
        )
    with (DATA / "bridge.csv").open() as f:
        br_rows = list(csv.DictReader(f))
    br_jw_ids = {r["source_id"] for r in br_rows if r["survey"] == "JWST"}
    n_br_sd = sum(1 for r in br_rows if r["survey"] == "SDSS")
    n_br_eu = sum(1 for r in br_rows if r["survey"] == "Euclid")

    with (DATA / "sdss.csv").open() as f:
        sd185 = list(csv.DictReader(f))
    with (DATA / "sdss.csv").open() as f:
        sd_z_map = {r["source_id"]: float(r["z"]) for r in csv.DictReader(f)}

    with (DATA / "euclid.csv").open() as f:
        eu334 = list(csv.DictReader(f))

    c_jwst = "#9E1F3A"
    c_sdss = "#1F5E8C"
    c_eucl = "#2B7A68"
    c_bridge = "#F5E7C4"

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.75), sharey=True)
    plt.subplots_adjust(wspace=0.07)

    xlim = (41.52, 44.05)
    ylim = (2.88, 4.19)

    for idx, ax in enumerate(axes):
        ax.axvspan(43.0, 43.7, color=c_bridge, alpha=0.52, zorder=0)
        ax.grid(color="#EEEEEE", lw=0.6, zorder=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\log_{10}\,(L_{\mathrm{H}\alpha}\ /\ \mathrm{erg\,s^{-1}})$")
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    # --- Panel (a): JWST (simplified markers: 20 primary colored, 20 other clean pale, 3 overlap rings) ---
    ax = axes[0]
    jw_x = np.array([float(r["h"]) for r in jw40])
    jw_y = np.array([math.log10(float(r["FWHM_sg_kms"])) for r in jw40])
    jw_z = [float(r["z_fit"]) for r in jw40]
    is_20 = np.array([(r["dedup_id"] in jw20_ids or r["canonical_name"] in jw20_ids) for r in jw40])
    is_other = ~is_20
    is_br = np.array([r["dedup_id"] in br_jw_ids for r in jw40])

    n_jw40 = len(jw40)
    n_jw20 = int(is_20.sum())
    n_jw_other = int(is_other.sum())
    n_br_jw = int(is_br.sum())

    sl40, in40, sd40 = _fit_ols_line(jw_x, jw_y)
    xg40 = np.linspace(jw_x.min(), jw_x.max(), 60)
    ax.plot(xg40, sl40 * xg40 + in40, color="#3A3A3A", ls="-", lw=1.55, zorder=2)

    ax.scatter(
        jw_x[is_other], jw_y[is_other], s=28, facecolors="#F4DFE4", edgecolors=c_jwst, lw=0.85,
        label=f"Other continuum-clean ({n_jw_other})", zorder=3
    )
    ax.scatter(
        jw_x[is_20], jw_y[is_20], s=36, facecolors=c_jwst, edgecolors="white", lw=0.45,
        label=f"Primary window sample ({n_jw20})", zorder=4
    )
    ax.scatter(
        jw_x[is_br], jw_y[is_br], s=98, facecolors="none", edgecolors="#111111", lw=1.25,
        label=f"Overlap band ({n_br_jw})", zorder=5
    )

    ax.set_title(
        f"(a) JWST LRDs ($z = {min(jw_z):.2f}$–${max(jw_z):.2f}$)\n"
        f"$N = {n_jw40}$, $s = {sd40:.4f}$ dex",
        fontsize=9.2, pad=5
    )
    ax.set_ylabel(r"$\log_{10}\,(\mathrm{FWHM}_{\mathrm{sg}}\ /\ \mathrm{km\,s^{-1}})$")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=7.4, borderpad=0.30)

    # --- Panel (b): SDSS ---
    ax = axes[1]
    sd_x = np.array([float(r["h"]) for r in sd185])
    sd_y = np.array([math.log10(float(r["FWHM_sg"])) for r in sd185])
    sd_z = [sd_z_map[r["source_id"]] for r in sd185]
    n_sd = len(sd185)
    sl_sd, in_sd, sd_sd = _fit_ols_line(sd_x, sd_y)
    xg_sd = np.linspace(sd_x.min(), sd_x.max(), 60)

    ax.scatter(sd_x, sd_y, s=18, color=c_sdss, alpha=0.68, edgecolors="none", zorder=3)
    ax.plot(xg_sd, sl_sd * xg_sd + in_sd, color="#123A58", ls="-", lw=1.6, zorder=4)
    ax.set_title(
        f"(b) SDSS Type-1 AGN ($z = {min(sd_z):.2f}$–${max(sd_z):.2f}$)\n"
        f"$N = {n_sd}$, $s = {sd_sd:.4f}$ dex",
        fontsize=9.2, pad=5
    )
    ax.text(
        43.35, 2.93, f"Overlap 43.0–43.7\n($N = {n_br_sd}$)",
        fontsize=7.4, color="#785E1E", ha="center", va="bottom"
    )

    # --- Panel (c): Euclid DR1 ---
    ax = axes[2]
    eu_x = np.array([float(r["h"]) for r in eu334])
    eu_y = np.array([float(r["w"]) for r in eu334])
    eu_z = [float(r["z"]) for r in eu334]
    n_eu = len(eu334)
    sl_eu, in_eu, sd_eu = _fit_ols_line(eu_x, eu_y)
    xg_eu = np.linspace(eu_x.min(), eu_x.max(), 60)

    ax.scatter(eu_x, eu_y, s=15, color=c_eucl, alpha=0.58, edgecolors="none", zorder=3)
    ax.plot(xg_eu, sl_eu * xg_eu + in_eu, color="#15473C", ls="-", lw=1.6, zorder=4)
    ax.set_title(
        f"(c) Euclid DR1 Primary ($z = {min(eu_z):.2f}$–${max(eu_z):.2f}$)\n"
        f"$N = {n_eu}$, $s = {sd_eu:.4f}$ dex",
        fontsize=9.2, pad=5
    )
    ax.text(
        43.35, 2.93, f"Overlap 43.0–43.7\n($N = {n_br_eu}$)",
        fontsize=7.4, color="#785E1E", ha="center", va="bottom"
    )

    ax_r = ax.twinx()
    ax_r.set_ylim(ylim)
    fwhm_ticks = [1000, 2000, 5000, 10000]
    ax_r.set_yticks([math.log10(v) for v in fwhm_ticks])
    ax_r.set_yticklabels(["1000", "2000", "5000", "10000"], fontsize=8.2)
    ax_r.set_ylabel(r"$\mathrm{FWHM}_{\mathrm{sg}}\ (\mathrm{km\,s^{-1}})$", fontsize=9.0)
    ax_r.tick_params(axis="y", direction="in", labelleft=False)
    ax.tick_params(axis="y", right=False)

    save_fig(fig, "fig3_three_survey_width_luminosity")


def make_figA1():
    """Appendix Figure A1: 6-window injection-recovery criteria and extended-window diagnostic."""
    with (TABLES / "calibration_summary.csv").open() as f:
        cal = {(r["group"], int(r["V"]), r["profile"]): r for r in csv.DictReader(f)}
    with (TABLES / "observed_scatter.csv").open() as f:
        obs = {(r["group"], int(r["V"])): r for r in csv.DictReader(f)}

    c_jwst = "#9E1F3A"
    c_sdss = "#1F5E8C"

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    plt.subplots_adjust(wspace=0.30)

    vs_all = [1500, 2500, 3500, 4500, 5500, 6000]
    vk_all = np.array(vs_all, dtype=float) / 1000.0

    # Panel (a): Precision criterion across 1500-6000 km/s
    ax0.axvspan(5.0, 6.35, color="#F2F2F2", zorder=0)
    ax0.grid(axis="y", color="#EAEAEA", lw=0.6, zorder=1)
    ax0.axhline(0.08, color="#555555", ls="--", lw=1.05, zorder=2)
    ax0.text(6.20, 0.0776, "0.08 dex\nthreshold", fontsize=7.3, color="#444444", va="top", ha="right", linespacing=0.95)

    for grp, col, label_base in (("JWST", c_jwst, "JWST"), ("SDSS", c_sdss, "SDSS")):
        for prof, ls, alpha, lw, mk in (("Gaussian", "-", 1.0, 1.65, "o"), ("variable wings", "--", 0.78, 1.25, "^")):
            errs = [float(cal[(grp, V, prof)]["error68"]) for V in vs_all]
            passed = [cal[(grp, V, prof)]["passed"] == "True" for V in vs_all]
            lbl = f"{label_base} ({'Gaussian' if prof == 'Gaussian' else 'core + wing'})"
            ax0.plot(vk_all, errs, color=col, ls=ls, lw=lw, alpha=alpha, label=lbl, zorder=3)
            for x, y, ok in zip(vk_all, errs, passed):
                ax0.plot(
                    x, y, marker=mk, ms=4.9 if prof == "Gaussian" else 4.4,
                    color=col, mfc=col if ok else "white", mew=1.15, alpha=alpha, zorder=4
                )

    open_proxy = mpl.lines.Line2D([0], [0], marker="o", ms=4.7, mfc="white", mec="#444444", ls="none", mew=1.1)
    handles, labels = ax0.get_legend_handles_labels()
    handles.append(open_proxy)
    labels.append(r"Open: fails $\geq 1$ criterion")

    ax0.set_xlim(1.25, 6.28)
    ax0.set_ylim(0.0, 0.099)
    ax0.set_xticks(vk_all)
    ax0.set_xticklabels(["1.5", "2.5", "3.5", "4.5", "5.5", "6.0"])
    ax0.set_xlabel(r"Window half-width $V\ (10^3\ \mathrm{km\,s^{-1}})$")
    ax0.set_ylabel(r"68th pct. $|\Delta\log_{10}\sigma(V)|$ (dex)")
    ax0.set_title("(a) Recovery error vs. window half-width", pad=6)
    ax0.legend(handles, labels, loc="upper left", frameon=True, framealpha=0.95,
               edgecolor="#CCCCCC", handlelength=1.8, fontsize=7.3, borderpad=0.30)

    # Panel (b): Extended-window diagnostic
    n_jwst = int(obs[("JWST", 2500)]["n"])
    sd_ns = [int(obs[("SDSS", V)]["n"]) for V in vs_all]
    n_sd_min, n_sd_max = min(sd_ns), max(sd_ns)

    ax1.axvspan(5.0, 6.35, color="#F2F2F2", zorder=0)
    ax1.text(5.65, 0.017, "JWST window\ndisqualified", fontsize=7.3, color="#666666", ha="center", va="bottom")

    jw_s = np.array([float(obs[("JWST", V)]["s"]) for V in vs_all])
    jw_lo = np.array([float(obs[("JWST", V)]["lo"]) for V in vs_all])
    jw_hi = np.array([float(obs[("JWST", V)]["hi"]) for V in vs_all])
    ax1.fill_between(vk_all[:4], jw_lo[:4], jw_hi[:4], color=c_jwst, alpha=0.14, lw=0, zorder=1)
    ax1.plot(
        vk_all[:4], jw_s[:4], color=c_jwst, ls="-", lw=1.85, marker="o", ms=5.0,
        label=f"JWST ({n_jwst}, validated)", zorder=4
    )
    ax1.errorbar(
        vk_all[4:], jw_s[4:], yerr=[jw_s[4:] - jw_lo[4:], jw_hi[4:] - jw_s[4:]],
        fmt="o", ms=5.0, color=c_jwst, mfc="white", mew=1.2, elinewidth=0.95, capsize=2.0, alpha=0.75,
        label=r"JWST diagnostic ($V \geq 5500$)", zorder=4
    )

    sd_s = np.array([float(obs[("SDSS", V)]["s"]) for V in vs_all])
    sd_lo = np.array([float(obs[("SDSS", V)]["lo"]) for V in vs_all])
    sd_hi = np.array([float(obs[("SDSS", V)]["hi"]) for V in vs_all])
    ax1.fill_between(vk_all, sd_lo, sd_hi, color=c_sdss, alpha=0.13, lw=0, zorder=1)
    ax1.plot(
        vk_all, sd_s, color=c_sdss, ls="--", lw=1.75, marker="s", ms=4.6,
        label=f"SDSS ({n_sd_min}–{n_sd_max}, default)", zorder=3
    )

    ax1.set_xlim(1.25, 6.28)
    ax1.set_ylim(0.012, 0.156)
    ax1.set_xticks(vk_all)
    ax1.set_xticklabels(["1.5", "2.5", "3.5", "4.5", "5.5", "6.0"])
    ax1.set_xlabel(r"Window half-width $V\ (10^3\ \mathrm{km\,s^{-1}})$")
    ax1.set_ylabel(r"Observed residual scatter $s(V)$ (dex)")
    ax1.set_title("(b) Extended-window scatter diagnostic", pad=6)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=7.4, borderpad=0.32)
    ax1.grid(axis="y", color="#EAEAEA", lw=0.6)

    save_fig(fig, "figA1_moment_calibration")


def make_figB1():
    """Appendix Figure B1: Compact single-column 33-source template amplitude breakdown."""
    with (TABLES / "template_groups.json").open() as f:
        tg = json.load(f)

    n_all = int(tg["all33"]["n"])
    n_top = int(tg["top3"]["n"])
    wt_top = 100.0 * float(tg["top3"]["archived_weight"])
    n_oth = int(tg["other30"]["n"])
    wt_oth = 100.0 * float(tg["other30"]["archived_weight"])

    fig, ax = plt.subplots(figsize=(5.1, 2.05))
    ax.axvline(1.0, color="#444444", ls="--", lw=1.0, zorder=1)
    ax.text(1.03, 2.48, "Equal amplitudes", fontsize=8.0, color="#444444", va="top")

    groups = [
        ("all33", f"All {n_all} sources (100% wt)", "#1F5E8C", "o"),
        ("top3", f"Top {n_top} sources ({wt_top:.1f}% wt)", "#9E1F3A", "D"),
        ("other30", f"Remaining {n_oth} ({wt_oth:.1f}% wt)", "#6E6E6E", "s"),
    ]
    y_pos = [2.0, 1.0, 0.0]
    for y, (gkey, glbl, col, mk) in zip(y_pos, groups):
        g = tg[gkey]
        val = float(g["wing_over_core"])
        lo, hi = float(g["ci95_pixel_conditional"][0]), float(g["ci95_pixel_conditional"][1])
        ax.plot([lo, hi], [y, y], color=col, lw=2.3, solid_capstyle="round", zorder=3)
        ax.plot(val, y, marker=mk, ms=6.0, color=col, mec="white", mew=0.6, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([g[1] for g in groups], fontsize=8.8)
    ax.set_xlim(-0.20, 2.15)
    ax.set_ylim(-0.55, 2.65)
    ax.set_xlabel("Relative wing amplitude", fontsize=9.0)
    ax.grid(axis="x", color="#EEEEEE", lw=0.6)
    ax.tick_params(axis="y", length=0, pad=5)

    save_fig(fig, "figB1_template_weights")


if __name__ == "__main__":
    set_apj_style()
    make_fig1()
    make_fig2()
    make_fig3()
    make_figA1()
    make_figB1()
