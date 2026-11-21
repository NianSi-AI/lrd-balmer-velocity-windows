import csv
import json
import math
import re
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "build/tables"
def read(path):
    return json.loads(path.read_text())


def _ols_ddof1(xs, ys):
    import numpy as np

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return {
        "n": len(x),
        "slope": float(slope),
        "intercept": float(intercept),
        "sd": float(np.std(resid, ddof=1)),
        "x_min": float(x.min()),
        "x_med": float(np.median(x)),
        "x_max": float(x.max()),
        "y_med": float(np.median(y)),
        "fwhm_med": float(10.0 ** np.median(y)),
    }


def build_supplement_values(base_values):
    """Build versioned supplementary macros from audited Phase 1, Phase 2, and frozen tables."""
    import numpy as np
    supp = {}

    def snum(name, value, decimals=3):
        assert name.isascii() and name.isalpha(), f"Invalid macro name: {name}"
        assert name not in base_values, f"Macro collision with frozen numbers.tex: {name}"
        assert name not in supp, f"Duplicate supplement macro: {name}"
        assert math.isfinite(value), f"Non-finite value for {name}: {value}"
        supp[name] = format(value, f".{decimals}f")

    def snum_signed(name, value, decimals=4):
        assert name.isascii() and name.isalpha(), f"Invalid macro name: {name}"
        assert name not in base_values and name not in supp
        assert math.isfinite(value)
        rounded = round(value, decimals)
        if rounded == 0.0:
            supp[name] = f"{0.0:.{decimals}f}"
        else:
            sign = "+" if rounded > 0 else "-"
            supp[name] = f"{sign}{abs(rounded):.{decimals}f}"

    # --- 1. Uniform ddof=1 integrated-width stats & sample descriptors across three surveys ---
    with (DATA / "jwst.csv").open() as f:
        jw_all = list(csv.DictReader(f))
    assert len(jw_all) == 43
    jw_map = {r["dedup_id"]: r for r in jw_all}
    jw40 = [r for r in jw_all if r["cont_qc_pass"] == "True"]
    jw40_ids = {r["dedup_id"] for r in jw40}
    with (TABLES / "edge_sensitivity_measurements.csv").open() as f:
        p2_edge_rows = list(csv.DictReader(f))
    p2_jw22_ids = sorted(set(r["source"] for r in p2_edge_rows if r["survey"] == "JWST"))
    jw22 = [r for r in jw40 if r["dedup_id"] in p2_jw22_ids or r["canonical_name"] in p2_jw22_ids]
    with (TABLES / "measurements.csv").open() as f:
        meas_rows_all = list(csv.DictReader(f))
    p2_jw20_ids = {
        r["source"] for r in meas_rows_all
        if r["survey"] == "JWST" and float(r["snr"]) >= 15.0 and int(float(r["V"])) == 2500 and r["fit_ok"] == "True"
    }
    jw20 = [r for r in jw40 if r["dedup_id"] in p2_jw20_ids or r["canonical_name"] in p2_jw20_ids]
    assert len(jw40) == 40 and len(jw22) == 22 and len(jw20) == 20

    with (DATA / "jwst.csv").open() as f:
        inv_map = {r["dedup_id"]: r for r in csv.DictReader(f)}
    p1_43 = sum(1 for r in jw_all if inv_map.get(r["dedup_id"], {}).get("in_paper1") == "True")
    p1_40 = sum(1 for r in jw40 if inv_map.get(r["dedup_id"], {}).get("in_paper1") == "True")
    p1_22 = sum(1 for r in jw22 if inv_map.get(r["dedup_id"], {}).get("in_paper1") == "True")
    p1_20 = sum(1 for r in jw20 if inv_map.get(r["dedup_id"], {}).get("in_paper1") == "True")
    atlas_40 = sum(
        1 for r in jw40
        if inv_map.get(r["dedup_id"], {}).get("in_yanagisawa26") == "True"
        or "atlas" in inv_map.get(r["dedup_id"], {}).get("provenances", "").lower()
    )
    bx_40 = sum(1 for r in jw40 if "baccus" in inv_map.get(r["dedup_id"], {}).get("provenances", "").lower())

    hb_ids = sorted(r["source"] for r in csv.DictReader((DATA / "template_sources.csv").read_text().splitlines()))
    assert len(hb_ids) == 33
    hb_in_clean = sum(1 for k in hb_ids if k in jw40_ids)
    hb_out_clean = sum(1 for k in hb_ids if k not in jw40_ids)
    hb_p1 = sum(1 for k in hb_ids if inv_map.get(k, {}).get("in_paper1") == "True")
    hb_z = [float(jw_map[k]["z_fit"]) for k in hb_ids]

    jw40_ols = _ols_ddof1([float(r["h"]) for r in jw40], [math.log10(float(r["FWHM_sg_kms"])) for r in jw40])
    jw22_ols = _ols_ddof1([float(r["h"]) for r in jw22], [math.log10(float(r["FWHM_sg_kms"])) for r in jw22])
    jw40_z = [float(r["z_fit"]) for r in jw40]
    jw22_z = [float(r["z_fit"]) for r in jw22]
    jw20_z = [float(r["z_fit"]) for r in jw20]

    snum("JWSTGratingParentN", len(jw_all), 0)
    snum("JWSTGratingParentPaperOneOverlapN", p1_43, 0)
    snum("JWSTCleanPaperOneOverlapN", p1_40, 0)
    snum("JWSTCleanAtlasProvN", atlas_40, 0)
    snum("JWSTCleanBXProvN", bx_40, 0)
    snum("JWSTPresetPaperOneOverlapN", p1_20, 0)
    snum("JWSTEdgeFallbackN", len(jw22) - len(jw20), 0)
    snum("JWSTOtherCleanN", len(jw40) - len(jw20), 0)
    snum("JWSTPresetZMin", min(jw20_z), 2)
    snum("JWSTPresetZMed", float(np.median(jw20_z)), 2)
    snum("JWSTPresetZMax", max(jw20_z), 2)
    snum("JWSTTwoLineInCleanN", hb_in_clean, 0)
    snum("JWSTTwoLineOutCleanN", hb_out_clean, 0)
    snum("JWSTTwoLinePaperOneOverlapN", hb_p1, 0)
    snum("JWSTTwoLineZMin", min(hb_z), 2)
    snum("JWSTTwoLineZMax", max(hb_z), 2)
    snum("JWSTCleanZMin", min(jw40_z), 2)
    snum("JWSTCleanZMed", float(np.median(jw40_z)), 2)
    snum("JWSTCleanZMax", max(jw40_z), 2)
    snum("JWSTCleanSGScatterUniform", jw40_ols["sd"], 4)
    snum("JWSTCleanSGSlope", jw40_ols["slope"], 3)
    snum("JWSTCleanMedFWHM", jw40_ols["fwhm_med"], 0)
    snum("JWSTCleanLHaMin", jw40_ols["x_min"], 2)
    snum("JWSTCleanLHaMed", jw40_ols["x_med"], 2)
    snum("JWSTCleanLHaMax", jw40_ols["x_max"], 2)

    snum("JWSTHighSNRN", len(jw22), 0)
    snum("JWSTHighSNRPaperOneOverlapN", p1_22, 0)
    snum("JWSTHighSNRZMin", min(jw22_z), 2)
    snum("JWSTHighSNRZMed", float(np.median(jw22_z)), 2)
    snum("JWSTHighSNRZMax", max(jw22_z), 2)
    snum("JWSTHighSNRSGScatterUniform", jw22_ols["sd"], 4)
    snum("JWSTHighSNRSGSlope", jw22_ols["slope"], 3)
    snum("JWSTHighSNRMedFWHM", jw22_ols["fwhm_med"], 0)
    snum("JWSTHighSNRLHaMin", jw22_ols["x_min"], 2)
    snum("JWSTHighSNRLHaMed", jw22_ols["x_med"], 2)
    snum("JWSTHighSNRLHaMax", jw22_ols["x_max"], 2)

    with (DATA / "sdss.csv").open() as f:
        sd185 = list(csv.DictReader(f))
    assert len(sd185) == 185
    with (DATA / "sdss.csv").open() as f:
        sd_z_map = {r["source_id"]: float(r["z"]) for r in csv.DictReader(f)}
    sd_z = [sd_z_map[r["source_id"]] for r in sd185]
    sd_ols = _ols_ddof1([float(r["h"]) for r in sd185], [math.log10(float(r["FWHM_sg"])) for r in sd185])

    snum("SDSSCleanZMin", min(sd_z), 2)
    snum("SDSSCleanZMed", float(np.median(sd_z)), 2)
    snum("SDSSCleanZMax", max(sd_z), 2)
    snum("SDSSCleanSGScatterUniform", sd_ols["sd"], 4)
    snum("SDSSCleanSGSlope", sd_ols["slope"], 3)
    snum("SDSSCleanMedFWHM", sd_ols["fwhm_med"], 0)
    snum("SDSSCleanLHaMin", sd_ols["x_min"], 2)
    snum("SDSSCleanLHaMed", sd_ols["x_med"], 2)
    snum("SDSSCleanLHaMax", sd_ols["x_max"], 2)

    with (DATA / "euclid.csv").open() as f:
        eu334 = list(csv.DictReader(f))
    assert len(eu334) == 334
    eu_z = [float(r["z"]) for r in eu334]
    eu_ols = _ols_ddof1([float(r["h"]) for r in eu334], [float(r["w"]) for r in eu334])

    snum("EuclidMainZMin", min(eu_z), 2)
    snum("EuclidMainZMed", float(np.median(eu_z)), 2)
    snum("EuclidMainZMax", max(eu_z), 2)
    snum("EuclidMainSGScatterUniform", eu_ols["sd"], 4)
    snum("EuclidMainSGSlope", eu_ols["slope"], 3)
    snum("EuclidMainMedFWHM", eu_ols["fwhm_med"], 0)
    snum("EuclidMainLHaMin", eu_ols["x_min"], 2)
    snum("EuclidMainLHaMed", eu_ols["x_med"], 2)
    snum("EuclidMainLHaMax", eu_ols["x_max"], 2)
    eu_jw_c_gap = float(base_values["EuclidBridgeCMin"]) - float(base_values["JWSTBridgeCMax"])
    snum("BridgeJWSTEuclidContinuumGap", eu_jw_c_gap, 3)

    # --- 2. Phase 2 cumulative second-moment results ---
    v_tags = {
        1500: "OneFive",
        2500: "TwoFive",
        3500: "ThreeFive",
        4500: "FourFive",
        5500: "FiveFive",
        6000: "SixZero",
    }
    with (TABLES / "observed_scatter.csv").open() as f:
        obs_rows = list(csv.DictReader(f))
    obs_map = {(r["group"], int(r["V"])): r for r in obs_rows}

    # JWST preset (N=20)
    snum("PTJWSTPresetN", int(obs_map[("JWST", 2500)]["n"]), 0)
    snum("PTJWSTPresetSGScatter", float(obs_map[("JWST", 2500)]["s_single_gauss"]), 4)
    for V, tag in v_tags.items():
        r = obs_map[("JWST", V)]
        snum(f"PTJWSTPresetS{tag}", float(r["s"]), 4)
        snum(f"PTJWSTPresetSLo{tag}", float(r["lo"]), 4)
        snum(f"PTJWSTPresetSHi{tag}", float(r["hi"]), 4)
        snum(f"PTJWSTPresetMedSig{tag}", float(r["median_sigma"]), 0)
    r25 = obs_map[("JWST", 2500)]
    r45 = obs_map[("JWST", 4500)]
    snum("PTJWSTPresetDelta", float(r45["delta"]), 4)
    snum_signed("PTJWSTPresetDeltaSigned", float(r45["delta"]), 4)
    snum("PTJWSTPresetDeltaLo", float(r45["delta_lo"]), 4)
    snum("PTJWSTPresetDeltaHi", float(r45["delta_hi"]), 4)
    snum("PTJWSTPresetSTwoFiveThreeDec", float(r25["s"]), 3)
    snum("PTJWSTPresetSFourFiveThreeDec", float(r45["s"]), 3)
    snum("PTJWSTPresetDeltaThreeDec", float(r45["delta"]), 3)
    snum_signed("PTJWSTPresetDeltaSignedThreeDec", float(r45["delta"]), 3)
    snum("PTJWSTPresetDeltaLoThreeDec", float(r45["delta_lo"]), 3)
    snum("PTJWSTPresetDeltaHiThreeDec", float(r45["delta_hi"]), 3)
    r35 = obs_map[("JWST", 3500)]
    snum("PTJWSTPresetDeltaThreeFive", float(r35["delta"]), 4)

    # Raw total (un-deblended) H-alpha check for JWST preset 20
    with (TABLES / "measurements.csv").open() as f:
        meas_rows = list(csv.DictReader(f))
    raw_tot = {}
    for V in (1500, 2500, 3500, 4500):
        sub = [r for r in meas_rows if r["survey"] == "JWST" and float(r["snr"]) >= 15.0 and int(float(r["V"])) == V and r["fit_ok"] == "True"]
        assert len(sub) == 20
        raw_tot[V] = _ols_ddof1([float(r["h"]) for r in sub], [math.log10(float(r["sigma_total"])) for r in sub])["sd"]
        snum(f"PTJWSTPresetRawTotS{v_tags[V]}", raw_tot[V], 4)
    snum("PTJWSTPresetRawTotDelta", raw_tot[4500] - raw_tot[2500], 4)

    # SDSS preset (N=169..167)
    for V, tag in v_tags.items():
        r = obs_map[("SDSS", V)]
        snum(f"PTSDSSPresetN{tag}", int(r["n"]), 0)
        snum(f"PTSDSSPresetS{tag}", float(r["s"]), 4)
        snum(f"PTSDSSPresetSLo{tag}", float(r["lo"]), 4)
        snum(f"PTSDSSPresetSHi{tag}", float(r["hi"]), 4)
        snum(f"PTSDSSPresetMedSig{tag}", float(r["median_sigma"]), 0)
    rs45 = obs_map[("SDSS", 4500)]
    snum("PTSDSSPresetDelta", float(rs45["delta"]), 4)
    snum_signed("PTSDSSPresetDeltaSigned", float(rs45["delta"]), 4)
    snum("PTSDSSPresetDeltaThreeDec", float(rs45["delta"]), 3)
    snum_signed("PTSDSSPresetDeltaSignedThreeDec", float(rs45["delta"]), 3)
    snum("PTSDSSPresetDeltaLo", float(rs45["delta_lo"]), 4)
    snum("PTSDSSPresetDeltaHi", float(rs45["delta_hi"]), 4)

    # JWST 40 clean mother & 8 unflagged-absorption high-S/N checks
    snum("PTJWSTFortyNFourFive", int(obs_map[("JWST40", 4500)]["n"]), 0)
    snum("PTJWSTFortySTwoFive", float(obs_map[("JWST40", 2500)]["s"]), 4)
    snum("PTJWSTFortySFourFive", float(obs_map[("JWST40", 4500)]["s"]), 4)
    snum("PTJWSTUnflaggedN", int(obs_map[("JWST unflagged", 4500)]["n"]), 0)
    snum("PTJWSTUnflaggedSTwoFive", float(obs_map[("JWST unflagged", 2500)]["s"]), 4)
    snum("PTJWSTUnflaggedSFourFive", float(obs_map[("JWST unflagged", 4500)]["s"]), 4)
    snum("PTJWSTUnflaggedDelta", float(obs_map[("JWST unflagged", 4500)]["delta"]), 4)
    snum("PTJWSTUnflaggedDeltaLo", float(obs_map[("JWST unflagged", 4500)]["delta_lo"]), 4)
    snum("PTJWSTUnflaggedDeltaHi", float(obs_map[("JWST unflagged", 4500)]["delta_hi"]), 4)

    # Full JWST (N=22) & SDSS supplemented (N=181..179) from edge_scatter.csv
    with (TABLES / "edge_scatter.csv").open() as f:
        edge_rows = list(csv.DictReader(f))
    edge_map = {(r["group"], int(r["V"])): r for r in edge_rows}

    snum("PTJWSTFullN", int(edge_map[("JWST", 2500)]["n"]), 0)
    snum("PTJWSTFullSGScatter", float(edge_map[("JWST", 2500)]["s_single_gauss"]), 4)
    for V in (1500, 2500, 3500, 4500):
        tag = v_tags[V]
        rj = edge_map[("JWST", V)]
        snum(f"PTJWSTFullS{tag}", float(rj["s"]), 4)
        snum(f"PTJWSTFullSLo{tag}", float(rj["lo"]), 4)
        snum(f"PTJWSTFullSHi{tag}", float(rj["hi"]), 4)
        snum(f"PTJWSTFullMedSig{tag}", float(rj["median_sigma"]), 0)
        rs = edge_map[("SDSS", V)]
        snum(f"PTSDSSFullN{tag}", int(rs["n"]), 0)
        snum(f"PTSDSSFullS{tag}", float(rs["s"]), 4)
        snum(f"PTSDSSFullSLo{tag}", float(rs["lo"]), 4)
        snum(f"PTSDSSFullSHi{tag}", float(rs["hi"]), 4)
        snum(f"PTSDSSFullMedSig{tag}", float(rs["median_sigma"]), 0)

    rj45 = edge_map[("JWST", 4500)]
    snum("PTJWSTFullDelta", float(rj45["delta"]), 4)
    snum_signed("PTJWSTFullDeltaSigned", float(rj45["delta"]), 4)
    snum("PTJWSTFullDeltaLo", float(rj45["delta_lo"]), 4)
    snum("PTJWSTFullDeltaHi", float(rj45["delta_hi"]), 4)
    rs45_f = edge_map[("SDSS", 4500)]
    snum("PTSDSSFullDelta", float(rs45_f["delta"]), 4)
    snum_signed("PTSDSSFullDeltaSigned", float(rs45_f["delta"]), 4)
    snum("PTSDSSFullDeltaLo", float(rs45_f["delta_lo"]), 4)
    snum("PTSDSSFullDeltaHi", float(rs45_f["delta_hi"]), 4)

    # Cross-survey scatter difference (JWST 22 - SDSS 179..181)
    with (TABLES / "cross_survey_scatter_difference.csv").open() as f:
        diff_map = {int(r["V"]): r for r in csv.DictReader(f)}
    for V in (2500, 3500, 4500):
        tag = v_tags[V]
        rd = diff_map[V]
        snum_signed(f"PTDiff{tag}", float(rd["JWST_minus_SDSS"]), 4)
        snum_signed(f"PTDiffLo{tag}", float(rd["lo95"]), 4)
        snum_signed(f"PTDiffHi{tag}", float(rd["hi95"]), 4)

    # Source-conditioned mock predictions (full 22/179 & preset 20/167)
    for fname, prefix in (("edge_conditioned_predictions.csv", "PTMockFull"),
                          ("source_conditioned_predictions.csv", "PTMockPreset")):
        with (TABLES / fname).open() as f:
            pmap = {(r["group"], r["profile"]): r for r in csv.DictReader(f)}
        for grp, gtag in (("JWST", "JWST"), ("SDSS", "SDSS")):
            for prof, ptag in (("Gaussian", "Gauss"), ("variable wings", "Wing")):
                r = pmap[(grp, prof)]
                key = f"{prefix}{gtag}{ptag}"
                snum(key + "DeltaTrue", float(r["delta_true_med"]), 4)
                snum(key + "DeltaLo", float(r["predicted_delta_lo"]), 4)
                snum(key + "DeltaMed", float(r["predicted_delta_med"]), 4)
                snum(key + "DeltaHi", float(r["predicted_delta_hi"]), 4)
                snum(key + "TrueTwoFive", float(r["true_s2500_med"]), 4)
                snum(key + "TrueFourFive", float(r["true_s4500_med"]), 4)
                snum(key + "TrueFull", float(r["true_full_med"]), 4)

    # Primary calibration summary & Stage-1 trial denominators (Appendix Table & text)
    with (TABLES / "calibration_summary.csv").open() as f:
        cal_map = {(r["group"], int(r["V"]), r["profile"]): r for r in csv.DictReader(f)}
    for grp, gtag in (("JWST", "JWST"), ("SDSS", "SDSS"), ("JWST40", "JWSTForty")):
        for V, vtag in v_tags.items():
            for prof, ptag in (("Gaussian", "Gauss"), ("variable wings", "Wing")):
                r = cal_map[(grp, V, prof)]
                base = f"PTCal{gtag}{ptag}{vtag}"
                snum(base + "UsablePct", 100.0 * float(r["usable"]), 1)
                snum_signed(base + "Bias", float(r["bias"]), 3)
                snum(base + "ErrSixEight", float(r["error68"]), 3)

    with (TABLES / "injections.csv").open() as f:
        inj_rows = list(csv.DictReader(f))
    jw_inj_15 = [
        r for r in inj_rows
        if r["survey"] == "JWST" and float(r["snr"]) >= 15.0 and int(float(r["V"])) == 1500 and r["profile"] == "Gaussian"
    ]
    sd_inj_15 = [
        r for r in inj_rows
        if r["survey"] == "SDSS" and int(float(r["V"])) == 1500 and r["profile"] == "Gaussian"
    ]
    jw_inj_ok = sum(1 for r in jw_inj_15 if r["recovered"] not in ("", "nan", "None") and float(r["recovered"]) > 0)
    sd_inj_ok = sum(1 for r in sd_inj_15 if r["recovered"] not in ("", "nan", "None") and float(r["recovered"]) > 0)
    snum("PTCalJWSTAttemptedTargetsN", len({r["source"] for r in jw_inj_15}), 0)
    snum("PTCalSDSSAttemptedTargetsN", len({r["source"] for r in sd_inj_15}), 0)
    snum("PTCalJWSTGaussOneFiveOkTrials", jw_inj_ok, 0)
    snum("PTCalJWSTGaussOneFiveTotTrials", len(jw_inj_15), 0)
    snum("PTCalSDSSGaussOneFiveOkTrials", sd_inj_ok, 0)
    snum("PTCalSDSSGaussOneFiveTotTrials", len(sd_inj_15), 0)

    # Edge-source individual recovery at V=4500 (Appendix)
    with (TABLES / "source_conditioned_injections.csv").open() as f:
        sci_rows = list(csv.DictReader(f))
    for dedup_id, tag in (("JWST_DEDUP_0039", "EdgeFiveZeroZeroFiveTwo"),
                          ("JWST_DEDUP_0055", "EdgeThreeEightFiveSixTwo")):
        for prof, ptag in (("Gaussian", "Gauss"), ("variable wings", "Wing")):
            sub = [r for r in sci_rows if r["source"] == dedup_id and int(float(r["V"])) == 4500 and r["profile"] == prof]
            ok = [r for r in sub if r["recovered"] not in ("", "nan", "None") and float(r["recovered"]) > 0]
            errs = [abs(math.log10(float(r["recovered"])) - math.log10(float(r["truth"]))) for r in ok]
            snum(f"PT{tag}{ptag}UsablePct", 100.0 * len(ok) / len(sub), 1)
            snum(f"PT{tag}{ptag}ErrSixEight", float(np.percentile(errs, 68)), 2)

    # --- 3. Phase 1 exact pixel-overlap Balmer windows & archived template ratio ---
    p1_meta = read(DATA / "balmer_sources.json")
    src_tags = {
        "RUBIES-EGS-42046": "SrcFourTwo",
        "RUBIES-EGS-55604": "SrcFiveFive",
        "JADES-GN-68797": "SrcSixEight",
    }
    for sname, stag in src_tags.items():
        m = p1_meta[sname]
        snum(f"POne{stag}ZOIII", float(m["z_OIII"]), 4)
        snum(f"POne{stag}FWHMsg", float(m["FWHM_sg_v3_kms"]), 0)
        ncomb = m["ncombine"]
        snum(f"POne{stag}NCombine", ncomb, 0)

    with (TABLES / "three_sources_observed_native_windows.csv").open() as f:
        modA = {(r["source"], r["window"], r["side"]): r for r in csv.DictReader(f)}
    with (TABLES / "total_unsubtracted_balmer.csv").open() as f:
        modB = {(r["source"], int(r["v_min_kms"]), int(r["v_max_kms"])): r for r in csv.DictReader(f)}

    win_specs = [
        ("center_lt300", 0, 300, "ZeroThree"),
        ("core_norm_300_1000", 300, 1000, "ThreeTen"),
        ("inner_1000_1500", 1000, 1500, "TenOneFive"),
        ("shoulder_1500_2500", 1500, 2500, "OneFiveTwoFive"),
        ("wing_2500_3500", 2500, 3500, "TwoFiveThreeFive"),
    ]
    for sname, stag in src_tags.items():
        for wname, vmin, vmax, wtag in win_specs:
            ra = modA[(sname, wname, "BOTH")]
            fa, efa = float(ra["F_Ha_1e17"]), float(ra["e_Ha_pixel_1e17"])
            fb, efb = float(ra["F_Hb_1e17"]), float(ra["e_Hb_pixel_1e17"])
            r_val = fb / fa
            r_err = math.sqrt((efb / fa) ** 2 + (fb * efa / (fa ** 2)) ** 2)
            snum(f"POneA{stag}{wtag}R", r_val, 3)
            snum(f"POneA{stag}{wtag}RErr", r_err, 3)
            if ra["D_point_if_Hb_ge_2_pixel_sigma"] != "":
                snum(f"POneA{stag}{wtag}D", float(ra["D_point_if_Hb_ge_2_pixel_sigma"]), 1)

            rb = modB[(sname, vmin, vmax)]
            fa_b, efa_b = float(rb["total_F_Ha_1e17"]), float(rb["e_Ha_1e17"])
            fb_b, efb_b = float(rb["total_F_Hb_1e17"]), float(rb["e_Hb_1e17"])
            rb_val = fb_b / fa_b
            rb_err = math.sqrt((efb_b / fa_b) ** 2 + (fb_b * efa_b / (fa_b ** 2)) ** 2)
            snum(f"POneB{stag}{wtag}R", rb_val, 3)
            snum(f"POneB{stag}{wtag}RErr", rb_err, 3)
            if rb["total_D_if_Hb_ge_2sigma"] != "":
                snum(f"POneB{stag}{wtag}D", float(rb["total_D_if_Hb_ge_2sigma"]), 1)

    # 42046 blue vs red 300-1000 asymmetry
    snum("POneASrcFourTwoThreeTenBlueD", float(modA[("RUBIES-EGS-42046", "core_norm_300_1000", "BLUE")]["D_point_if_Hb_ge_2_pixel_sigma"]), 1)
    snum("POneASrcFourTwoThreeTenRedD", float(modA[("RUBIES-EGS-42046", "core_norm_300_1000", "RED")]["D_point_if_Hb_ge_2_pixel_sigma"]), 1)

    # Archived 33-source / Top-3 / Other-30 template operator ratios
    tg = read(TABLES / "template_groups.json")
    for gkey, gtag in (("all33", "AllThirtyThree"), ("top3", "TopThree"), ("other30", "OtherThirty")):
        g = tg[gkey]
        snum(f"POneTmpl{gtag}N", int(g["n"]), 0)
        snum(f"POneTmpl{gtag}WtPct", 100.0 * float(g["archived_weight"]), 1)
        snum(f"POneTmpl{gtag}Ratio", float(g["wing_over_core"]), 3)
        snum(f"POneTmpl{gtag}Lo", float(g["ci95_pixel_conditional"][0]), 3)
        snum(f"POneTmpl{gtag}Hi", float(g["ci95_pixel_conditional"][1]), 3)

    return supp



def main():
    def catalog(name):
        with (DATA / name).open() as stream:
            return list(csv.DictReader(stream))

    summary = read(DATA / "survey_summary.json")
    bridge = catalog("bridge.csv")
    continuum = {survey:[float(row["log_lambda_Llambda_erg_s"]) for row in bridge
                         if row["survey"] == survey] for survey in ("JWST","SDSS","Euclid")}
    q = summary["euclid_sigma_w_quantiles"]
    base = {
        "EuclidMainN": str(len(catalog("euclid.csv"))),
        "EuclidHLimit": format(summary["euclid_h_limit"],".2f"),
        "EuclidTargets": str(summary["euclid_parent_n"]),
        "EuclidMainSigmaW": format(q[2],".3f"),
        "EuclidMainSigmaWLowerNinetyFive": format(q[0],".3f"),
        "EuclidMainSigmaWUpperNinetyFive": format(q[-1],".3f"),
        "SDSSSpectra":str(summary["sdss_attempted"]),
        "SDSSCleanN":str(len(catalog("sdss.csv"))),
        "JWSTCleanN":str(sum(r["cont_qc_pass"]=="True" for r in catalog("jwst.csv"))),
        "EuclidBridgeCMin":format(min(continuum["Euclid"]),".3f"),
        "JWSTBridgeCMax":format(max(continuum["JWST"]),".3f"),
        **{survey+"BridgeN":str(len(rows)) for survey,rows in continuum.items()}
    }
    values = {**base, **build_supplement_values(base)}
    manuscript = (ROOT / "manuscript/main.tex").read_text()
    used = set(re.findall(r"\\([A-Za-z]+)", manuscript)) & values.keys()
    # Check all numerical macros used in the paper at its reported precision.
    canonical = "".join(k+"="+values[k]+"\n" for k in sorted(used))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if digest != "a40e7155d26979ec19dd02f5490c744e2558de0617afadfa2ddb3baf3f5b389f":
        raise ValueError(f"Numerical comparison with manuscript v3.1 failed: {digest}")
    (ROOT / "build/numbers.tex").write_text("".join(
        "\\newcommand{\\" + k + "}{" + values[k] + "}\n" for k in sorted(used)))
    print(f"{len(used)} manuscript quantities matched at published precision.")


if __name__ == "__main__":
    main()
