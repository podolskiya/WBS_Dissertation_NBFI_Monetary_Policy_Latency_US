"""
04_regression_p1.py

Proposition 1 local projections (Jorda, 2005): does bank-NBFI linkage intensity (L_it) alter the strength of monetary policy transmission to
bank credit growth, at horizons h = 0..8 quarters?

Outcome (h-quarter cumulative log change from t-1): Delta_h y_{i,t} = log(CI_{i,t+h}) - log(CI_{i,t-1})
Built via calendar-based merges, robust to panel gaps. Because the outcome is already cumulative from t-1, delta_h at h=8 IS the
cumulative eight-quarter effect. Peak effect = max|delta_h| across h; time-to-peak = the h at which that max occurs.

Model A (PRIMARY) | bank + quarter FE:
    Delta_h y_{i,t} = alpha_i + gamma_t + delta_h (mp_shock_t x L_it) + phi_h L_it + theta_h' X_it + u_{i,t+h}
    mp_shock_t is absorbed by quarter FE; only delta_h is estimable.

Model B (ROBUSTNESS) | bank FE only:
    Delta_h y_{i,t} = alpha_i + beta_h mp_shock_t + delta_h (mp_shock_t x L_it) + phi_h L_it + psi_h' Z_t + theta_h' X_it + u_{i,t+h}
    Z_t = nfci, term_spread. Both beta_h and delta_h estimable.

Model C (ROBUSTNESS) | bank + quarter FE, L_{i,t-1} instead of L_it.
Model D (ROBUSTNESS) | bank + quarter FE, time-averaged L_it ("linkage type").
Model E (EXTENSION)  | bank + quarter FE, adds shock x L_it x high_stress
                         (high_stress = NFCI top historical tercile).
Model F (EXTENSION)  | bank FE only, estimated separately within
                         Low/Medium/High linkage-type tercile subsamples.

INPUT   data/clean/panel_macro.parquet
OUTPUT  outputs/table_p1_local_projections.csv / .txt
        outputs/table_p1_stress_dependent.csv / .txt
        outputs/table_p1_linkage_groups.csv / .txt
        outputs/fig_p1_impulse_response.png
        outputs/fig_p1_stress_dependent.png
        outputs/fig_p1_linkage_groups.png
        outputs/p1_regression_log.txt
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from linearmodels.panel import PanelOLS
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
config = import_module("00_config")

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
NAVY, BLUE, RED, GRAY = "#003366", "#0066cc", "#cc0000", "#666666"
GREEN = "#006633"

HORIZONS = list(range(0, 9))
CONTROLS = ["capital_ratio", "log_assets", "ci_loan_growth"]
WINSOR_LO, WINSOR_HI = 0.01, 0.99
INTERACT_VAR = "shock_x_L"

def _build_horizon_outcomes(df):
    """Delta_h y_{i,t} = log(CI_{i,t+h}) - log(CI_{i,t-1}), h = 0..8, via
    calendar-based merges. Shift the MAIN row's own date to the target
    quarter and merge against an unshifted copy -- shifting the source
    copy instead silently reverses the direction (see README)."""
    df = df.sort_values(["rssd9001", "rssd9999"]).reset_index(drop=True)
    src = df[["rssd9001", "rssd9999", "ci_level"]].copy()

    df["rssd9999_target"] = df["rssd9999"] - pd.DateOffset(months=3)
    df["rssd9999_target"] = df["rssd9999_target"] + pd.offsets.QuarterEnd(0)
    lag_src = src.rename(columns={"rssd9999": "rssd9999_target", "ci_level": "ci_lag1"})
    df = df.merge(lag_src, on=["rssd9001", "rssd9999_target"], how="left")
    df = df.drop(columns=["rssd9999_target"])

    for h in HORIZONS:
        df["rssd9999_target"] = df["rssd9999"] + pd.DateOffset(months=3 * h)
        df["rssd9999_target"] = df["rssd9999_target"] + pd.offsets.QuarterEnd(0)
        fwd_src = src.rename(columns={"rssd9999": "rssd9999_target", "ci_level": f"ci_fwd{h}"})
        df = df.merge(fwd_src, on=["rssd9001", "rssd9999_target"], how="left")
        df = df.drop(columns=["rssd9999_target"])
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(df[f"ci_fwd{h}"] / df["ci_lag1"])
        raw = raw.replace([np.inf, -np.inf], np.nan)
        n_valid = raw.notna().sum()
        if n_valid > 10:
            lo, hi = raw.quantile([WINSOR_LO, WINSOR_HI])
            df[f"y_h{h}"] = raw.clip(lower=lo, upper=hi)
        else:
            df[f"y_h{h}"] = raw
    return df

def _drop_singletons(est: pd.DataFrame, drop_time_singletons: bool = True) -> pd.DataFrame:
    """Drop singleton entities/time periods before fitting an FE model --
    a singleton is perfectly absorbed by its own FE and corrupts the
    residual-df count linearmodels uses internally (ZeroDivisionError
    otherwise). Iterates since dropping one can create another."""
    prev_len = -1
    while len(est) != prev_len:
        prev_len = len(est)
        bank_counts = est["rssd9001"].value_counts()
        good_banks = bank_counts[bank_counts > 1].index
        est = est[est["rssd9001"].isin(good_banks)]
        if drop_time_singletons:
            q_counts = est["rssd9999"].value_counts()
            good_qs = q_counts[q_counts > 1].index
            est = est[est["rssd9999"].isin(good_qs)]
    return est

def _fit_bankquarter_fe(df, h, L_col, model_label):
    """Shared bank+quarter FE estimator, parameterized by which linkage
    variable enters the main effect and shock interaction (L_it for
    Model A, L_it_lag1 for C, L_it_bankmean for D)."""
    cols = [f"y_h{h}", L_col, "mp_shock"] + CONTROLS
    est = df[["rssd9001", "rssd9999"] + cols].dropna().copy()
    est["shock_x_L"] = est["mp_shock"] * est[L_col]
    n_before_singleton = len(est)
    est = _drop_singletons(est)
    n_singleton_dropped = n_before_singleton - len(est)
    n_banks = est["rssd9001"].nunique()
    est = est.set_index(["rssd9001", "rssd9999"])
    y = est[f"y_h{h}"]
    X = est[[L_col, INTERACT_VAR] + CONTROLS]
    try:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        n_clusters_bank = est.index.get_level_values(0).nunique()
        n_clusters_time = est.index.get_level_values(1).nunique()
        two_way_se = res.std_errors.get(INTERACT_VAR, np.nan)
        needs_fallback = pd.isna(two_way_se)
        fallback_reason = "NaN"
        if not needs_fallback:
            res_oneway_check = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            one_way_se = res_oneway_check.std_errors.get(INTERACT_VAR, np.nan)
            if not pd.isna(one_way_se) and one_way_se > 0 and two_way_se < 0.2 * one_way_se:
                needs_fallback = True
                fallback_reason = (f"deflated (two-way SE={two_way_se:.4f} is <20% of "
                                   f"one-way SE={one_way_se:.4f} -- likely degenerate "
                                   f"covariance, not genuine precision)")
        if needs_fallback:
            res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            res._cov_note = (f"FALLBACK: one-way (bank) clustered SE used -- two-way "
                             f"clustered covariance was {fallback_reason} at this horizon "
                             f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time})")
        else:
            res._cov_note = None
        return res, len(est), n_banks, n_singleton_dropped
    except Exception as e:
        print(f"{model_label} FAILED at h={h}: {e}")
        return None, len(est), n_banks, n_singleton_dropped


def _fit_model_A(df, h):
    """Bank + quarter FE, contemporaneous L_it. Primary specification."""
    return _fit_bankquarter_fe(df, h, "L_it", "Model A")


def _fit_model_B(df, h):
    """Bank FE only. Both mp_shock (beta_h) and interaction (delta_h) estimable."""
    cols = [f"y_h{h}", "L_it", "mp_shock", "nfci", "term_spread"] + CONTROLS
    est = df[["rssd9001", "rssd9999"] + cols].dropna().copy()
    est["shock_x_L"] = est["mp_shock"] * est["L_it"]
    est = _drop_singletons(est, drop_time_singletons=False)
    n_banks = est["rssd9001"].nunique()
    est = est.set_index(["rssd9001", "rssd9999"])
    y = est[f"y_h{h}"]
    X = est[["mp_shock", "L_it", INTERACT_VAR, "nfci", "term_spread"] + CONTROLS]
    try:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=False, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        n_clusters_bank = est.index.get_level_values(0).nunique()
        n_clusters_time = est.index.get_level_values(1).nunique()
        two_way_se = res.std_errors.get(INTERACT_VAR, np.nan)
        needs_fallback = pd.isna(two_way_se)
        fallback_reason = "NaN"
        if not needs_fallback:
            res_oneway_check = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            one_way_se = res_oneway_check.std_errors.get(INTERACT_VAR, np.nan)
            if not pd.isna(one_way_se) and one_way_se > 0 and two_way_se < 0.2 * one_way_se:
                needs_fallback = True
                fallback_reason = (f"deflated (two-way SE={two_way_se:.4f} is <20% of "
                                   f"one-way SE={one_way_se:.4f} -- likely degenerate "
                                   f"covariance, not genuine precision)")
        if needs_fallback:
            res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            res._cov_note = (f"FALLBACK: one-way (bank) clustered SE used -- two-way "
                             f"clustered covariance was {fallback_reason} at this horizon "
                             f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time})")
        else:
            res._cov_note = None
        return res, len(est), n_banks
    except Exception as e:
        print(f"Model B FAILED at h={h}: {e}")
        return None, len(est), n_banks

def _fit_model_C(df, h):
    return _fit_bankquarter_fe(df, h, "L_it_lag1", "Model C")

def _fit_model_D(df, h):
    return _fit_bankquarter_fe(df, h, "L_it_bankmean", "Model D")

def _build_alt_linkage_vars(df):
    """L_it_lag1 (calendar-based one-quarter lag, same merge pattern as
    the horizon outcomes) and L_it_bankmean (bank-level time average --
    time-invariant, so its main effect is absorbed by entity FE and only
    the interaction with mp_shock_t is estimable)."""
    df = df.sort_values(["rssd9001", "rssd9999"]).reset_index(drop=True)
    src = df[["rssd9001", "rssd9999", "L_it"]].copy()
    df["rssd9999_target"] = df["rssd9999"] - pd.DateOffset(months=3)
    df["rssd9999_target"] = df["rssd9999_target"] + pd.offsets.QuarterEnd(0)
    lag_src = src.rename(columns={"rssd9999": "rssd9999_target", "L_it": "L_it_lag1"})
    df = df.merge(lag_src, on=["rssd9001", "rssd9999_target"], how="left")
    df = df.drop(columns=["rssd9999_target"])
    df["L_it_bankmean"] = df.groupby("rssd9001")["L_it"].transform("mean")
    return df

def _build_stress_indicator(df, threshold_pct=2/3):
    unique_q = df[["rssd9999", "nfci"]].drop_duplicates(subset=["rssd9999"])
    threshold = unique_q["nfci"].quantile(threshold_pct)
    df["high_stress"] = (df["nfci"] >= threshold).astype(int)
    return df, threshold

def _fit_model_E(df, h):
    cols = [f"y_h{h}", "L_it", "mp_shock", "high_stress"] + CONTROLS
    est = df[["rssd9001", "rssd9999"] + cols].dropna().copy()
    est["shock_x_L"] = est["mp_shock"] * est["L_it"]
    est["L_x_stress"] = est["L_it"] * est["high_stress"]
    est["shock_x_L_x_stress"] = est["mp_shock"] * est["L_it"] * est["high_stress"]
    n_before_singleton = len(est)
    est = _drop_singletons(est)
    n_singleton_dropped = n_before_singleton - len(est)
    n_banks = est["rssd9001"].nunique()
    est = est.set_index(["rssd9001", "rssd9999"])
    y = est[f"y_h{h}"]
    X = est[["L_it", "L_x_stress", INTERACT_VAR, "shock_x_L_x_stress"] + CONTROLS]
    try:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        n_clusters_bank = est.index.get_level_values(0).nunique()
        n_clusters_time = est.index.get_level_values(1).nunique()
        if pd.isna(res.std_errors.get("shock_x_L_x_stress", np.nan)):
            res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            res._cov_note = (f"FALLBACK: one-way (bank) clustered SE used -- two-way "
                             f"clustered covariance was degenerate (NaN) at this horizon "
                             f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time})")
        else:
            res._cov_note = None
        return res, len(est), n_banks, n_singleton_dropped
    except Exception as e:
        print(f"Model E FAILED at h={h}: {e}")
        return None, len(est), n_banks, n_singleton_dropped

def _build_linkage_terciles(df):
    bank_level = df[["rssd9001", "L_it_bankmean"]].drop_duplicates(subset=["rssd9001"])
    q33, q67 = bank_level["L_it_bankmean"].quantile([1/3, 2/3])

    def _assign(x):
        if x <= q33:
            return "Low"
        elif x <= q67:
            return "Medium"
        return "High"

    bank_group_map = bank_level.set_index("rssd9001")["L_it_bankmean"].apply(_assign)
    df["linkage_group"] = df["rssd9001"].map(bank_group_map)
    return df, q33, q67

def _fit_model_F(df, h, group):
    sub = df[df["linkage_group"] == group]
    cols = [f"y_h{h}", "mp_shock", "nfci", "term_spread"] + CONTROLS
    est = sub[["rssd9001", "rssd9999"] + cols].dropna().copy()
    n_before_singleton = len(est)
    est = _drop_singletons(est, drop_time_singletons=False)
    n_singleton_dropped = n_before_singleton - len(est)
    n_banks = est["rssd9001"].nunique()
    est = est.set_index(["rssd9001", "rssd9999"])
    y = est[f"y_h{h}"]
    X = est[["mp_shock", "nfci", "term_spread"] + CONTROLS]
    try:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=False, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        n_clusters_bank = est.index.get_level_values(0).nunique()
        n_clusters_time = est.index.get_level_values(1).nunique()
        if pd.isna(res.std_errors.get("mp_shock", np.nan)):
            res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            res._cov_note = (f"FALLBACK: one-way (bank) clustered SE used, group={group}, h={h} "
                             f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time})")
        else:
            res._cov_note = None
        return res, len(est), n_banks, n_singleton_dropped
    except Exception as e:
        print(f"Model F ({group}) FAILED at h={h}: {e}")
        return None, len(est), n_banks, n_singleton_dropped

def _stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def main():
    log = [
        "P1 LOCAL PROJECTIONS AUDIT LOG", "=" * 65,
        f"Horizons: h = {HORIZONS}",
        "Model A (primary): bank + quarter FE, contemporaneous L_it, two-way clustered SE",
        "Model B (robustness): bank FE only + nfci/term_spread, contemporaneous L_it, two-way clustered SE",
        "Model C (robustness): bank + quarter FE, L_{i,t-1} (lagged linkage, avoids simultaneity)",
        "Model D (robustness): bank + quarter FE, time-averaged L_it (\"linkage type\", avoids flow-measure noise)",
        "Model E (extension): bank + quarter FE, stress-dependent triple interaction "
        "(shock x L_it x high_stress, NFCI top tercile)",
        "",
    ]

    df = pd.read_parquet(config.F_PANEL_MACRO)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])
    df = _build_horizon_outcomes(df)
    df = _build_alt_linkage_vars(df)
    df, stress_threshold = _build_stress_indicator(df)
    n_stress_q = df.loc[df["high_stress"] == 1, "rssd9999"].nunique()
    n_total_q = df["rssd9999"].nunique()
    log.append(f"High-stress threshold (NFCI >= top-tercile value {stress_threshold:.4f}): "
               f"{n_stress_q} of {n_total_q} quarters flagged")

    df, q33, q67 = _build_linkage_terciles(df)
    group_bank_counts = df.drop_duplicates(subset=["rssd9001"])["linkage_group"].value_counts()
    log.append(f"Linkage-type terciles: Low<={q33:.4f}, Medium<={q67:.4f}, High>{q67:.4f}; "
               f"bank counts: {group_bank_counts.to_dict()}")

    n_lag_valid = df["L_it_lag1"].notna().sum()
    log.append(f"L_it_lag1 valid obs: {n_lag_valid:,} of {len(df):,}")

    for h in HORIZONS:
        n_valid = df[f"y_h{h}"].notna().sum()
        log.append(f"  h={h}: {n_valid:,} valid outcome observations (of {len(df):,})")

    results_A, results_B, results_C, results_D, results_E = {}, {}, {}, {}, {}
    for h in HORIZONS:
        res_A, n_A, banks_A, dropped_A = _fit_model_A(df, h)
        res_B, n_B, banks_B = _fit_model_B(df, h)
        res_C, n_C, banks_C, dropped_C = _fit_model_C(df, h)
        res_D, n_D, banks_D, dropped_D = _fit_model_D(df, h)
        res_E, n_E, banks_E, dropped_E = _fit_model_E(df, h)
        results_A[h] = {"res": res_A, "n": n_A, "banks": banks_A}
        results_B[h] = {"res": res_B, "n": n_B, "banks": banks_B}
        results_C[h] = {"res": res_C, "n": n_C, "banks": banks_C}
        results_D[h] = {"res": res_D, "n": n_D, "banks": banks_D}
        results_E[h] = {"res": res_E, "n": n_E, "banks": banks_E}
        if dropped_A > 0:
            log.append(f"  h={h}: dropped {dropped_A} singleton obs before Model A (bank/quarter FE)")
        if dropped_C > 0:
            log.append(f"  h={h}: dropped {dropped_C} singleton obs before Model C")
        if dropped_D > 0:
            log.append(f"  h={h}: dropped {dropped_D} singleton obs before Model D")
        if dropped_E > 0:
            log.append(f"  h={h}: dropped {dropped_E} singleton obs before Model E")
        for label, res in [("Model A", res_A), ("Model B", res_B), ("Model C", res_C),
                           ("Model D", res_D), ("Model E", res_E)]:
            if res is not None and getattr(res, "_cov_note", None):
                log.append(f"  h={h} {label}: {res._cov_note}")

    rows = []
    for h in HORIZONS:
        for model_name, results in [("A", results_A), ("B", results_B), ("C", results_C), ("D", results_D)]:
            r = results[h]
            if r["res"] is None:
                rows.append({"model": model_name, "h": h, "coef": np.nan, "se": np.nan,
                            "pval": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                            "n": r["n"], "banks": r["banks"]})
                continue
            res = r["res"]
            coef = res.params[INTERACT_VAR]
            se = res.std_errors[INTERACT_VAR]
            pval = res.pvalues[INTERACT_VAR]
            ci = res.conf_int().loc[INTERACT_VAR]
            rows.append({
                "model": model_name, "h": h, "coef": coef, "se": se, "pval": pval,
                "ci_lo": ci["lower"], "ci_hi": ci["upper"],
                "n": r["n"], "banks": r["banks"], "stars": _stars(pval),
                "cov_note": getattr(res, "_cov_note", None),
            })
    df_csv = pd.DataFrame(rows)
    df_csv.to_csv(config.OUTPUT / "table_p1_local_projections.csv", index=False, float_format="%.6f")

    txt_lines = [
        "TABLE: P1 Local Projections -- delta_h coefficient on (shock x linkage)",
        "Outcome: Delta_h y_it = log(CI_{i,t+h}) - log(CI_{i,t-1})",
        f"Sample: FFIEC 031 banks | Horizons h = 0..{max(HORIZONS)} quarters",
        "",
        f"{'h':<4}{'A: contemp L_it':>18}{'B: contemp,no-t.FE':>20}"
        f"{'C: lagged L':>16}{'D: bank-mean L':>16}",
        "-" * 74,
    ]
    for h in HORIZONS:
        line = f"{h:<4}"
        for results in [results_A, results_B, results_C, results_D]:
            r = results[h]
            if r["res"] is not None:
                c = r["res"].params[INTERACT_VAR]
                p = r["res"].pvalues[INTERACT_VAR]
                dagger = "\u2020" if getattr(r["res"], "_cov_note", None) else ""
                s = f"{c:+.4f}{_stars(p)}{dagger}"
            else:
                s = "failed"
            line += f"{s:>18}" if results is results_A else f"{s:>16}"
        txt_lines.append(line)
    txt_lines += ["-" * 74, "", "N by horizon (Model A / B / C / D):"]
    for h in HORIZONS:
        ns = [str(results[h]["n"]) for results in [results_A, results_B, results_C, results_D]]
        txt_lines.append(f"  h={h}: " + " / ".join(ns))
    txt_lines += [
        "", "-" * 74, "",
        "\u2020 Two-way (bank, quarter) clustered SE was degenerate (NaN) at this horizon;",
        "  one-way (bank) clustered SE reported instead. See p1_regression_log.txt for",
        "  cluster counts. Coefficient (point estimate) is unaffected either way.",
        "",
        "Notes: Coefficient shown is delta_h, on (mp_shock_t x linkage variable). Two-way",
        "clustered SEs (bank, quarter) unless noted. *** p<0.01  ** p<0.05  * p<0.10.",
        "Model A (PRIMARY): bank + quarter FE, contemporaneous L_it. mp_shock_t is",
        "  absorbed by quarter FE; delta_h is the only estimable parameter of interest.",
        "Model B (robustness): bank FE only, + nfci/term_spread, contemporaneous L_it.",
        "  Literal proposal-spec check; both beta_h and delta_h estimable, weaker ID.",
        "Model C (robustness): bank + quarter FE, L_{i,t-1} instead of contemporaneous",
        "  L_it -- avoids simultaneity between the shock and the linkage measure within",
        "  the same quarter (Kashyap-Stein 1995, Kishan-Opiela 2000).",
        "Model D (robustness): bank + quarter FE, time-averaged L_it (\"linkage type\")",
        "  instead of the noisy quarterly flow measure -- avoids attenuation from",
        "  quarter-to-quarter measurement noise (L_it has a large mass at exactly zero).",
        "mp_shock is a quarterly SUM of the Jarocinski-Karadi monthly MP_pm shock --",
        "  identifies the effect of the quarterly-cumulated shock, not a single FOMC",
        "  announcement (frequency-mismatch limitation, see 03_macro_merge.py).",
        "Because the outcome is cumulative from t-1, delta_h at h=8 IS the cumulative",
        "  eight-quarter effect (no further summation needed).",
    ]
    (config.OUTPUT / "table_p1_local_projections.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    for label, results in [("A (primary)", results_A), ("B", results_B),
                           ("C (lagged L)", results_C), ("D (bank-mean L)", results_D)]:
        coefs = {h: results[h]["res"].params[INTERACT_VAR]
                 for h in HORIZONS if results[h]["res"] is not None}
        if not coefs:
            continue
        peak_h = max(coefs, key=lambda h: abs(coefs[h]))
        peak_val = coefs[peak_h]
        cum_8 = coefs.get(8, np.nan)
        summary_lines = [
            "", f"PEAK EFFECT / TIME-TO-PEAK / CUMULATIVE EFFECT (Model {label})",
            f"  Peak |delta_h|: h={peak_h}, delta_{peak_h}={peak_val:+.4f}",
            f"  Cumulative 8-quarter effect: delta_8={cum_8:+.4f}" if not np.isnan(cum_8)
                else "  Cumulative 8-quarter effect: NOT AVAILABLE (insufficient N at h=8)",
        ]
        log.extend(summary_lines)

    e_rows = []
    e_lines = [
        "TABLE: P1 Extension -- Stress-Dependent Triple Interaction (Model E)",
        f"High-stress = NFCI in top tercile (threshold={stress_threshold:.4f})",
        "Bank + quarter FE throughout. mp_shock, high_stress, and their product",
        "are absorbed by quarter FE; the four estimable terms are shown below.", "",
        f"{'h':<4}{'calm (shock x L)':>18}{'stress diff.':>16}{'total (stress)':>16}{'total SE':>10}", "-" * 68,
    ]
    for h in HORIZONS:
        res = results_E[h]["res"]
        if res is None:
            e_lines.append(f"{h:<4}{'failed':>18}")
            continue
        calm = res.params[INTERACT_VAR]
        calm_se = res.std_errors[INTERACT_VAR]
        calm_p = res.pvalues[INTERACT_VAR]
        stress_diff = res.params["shock_x_L_x_stress"]
        stress_diff_p = res.pvalues["shock_x_L_x_stress"]
        total = calm + stress_diff
        try:
            cov_matrix = res.cov
            var_total = (cov_matrix.loc[INTERACT_VAR, INTERACT_VAR]
                        + cov_matrix.loc["shock_x_L_x_stress", "shock_x_L_x_stress"]
                        + 2 * cov_matrix.loc[INTERACT_VAR, "shock_x_L_x_stress"])
            total_se = np.sqrt(var_total) if var_total > 0 else np.nan
        except Exception:
            total_se = np.nan
        if not np.isnan(total_se) and total_se > 0:
            total_tstat = total / total_se
            total_pval = 2 * (1 - stats.norm.cdf(np.abs(total_tstat)))
        else:
            total_pval = np.nan
        e_lines.append(
            f"{h:<4}{f'{calm:+.4f}{_stars(calm_p)}':>18}{f'{stress_diff:+.4f}{_stars(stress_diff_p)}':>16}"
            f"{f'{total:+.4f}{_stars(total_pval)}':>16}{total_se:>10.4f}"
        )
        e_rows.append({"h": h, "calm": calm, "calm_se": calm_se, "calm_pval": calm_p,
                       "stress_differential": stress_diff, "stress_differential_pval": stress_diff_p,
                       "total_high_stress": total, "total_high_stress_se": total_se,
                       "total_high_stress_pval": total_pval, "n": results_E[h]["n"]})
    e_lines += [
        "-" * 68, "",
        "Notes: 'calm' = coefficient on (mp_shock x L_it), i.e. the transmission-",
        "  heterogeneity effect during LOW-stress quarters (the baseline).",
        "'stress diff.' = coefficient on (mp_shock x L_it x high_stress), i.e. how much",
        "  MORE (or less) the effect is during HIGH-stress quarters.",
        "'total (stress)' = calm + stress diff. = the total effect during high-stress",
        "  quarters; its SE uses the joint covariance of the two coefficients, not",
        "  independent addition-in-quadrature.",
        "Motivated by the dissertation plan's own Brunnermeier-Pedersen funding-stress",
        "  channel discussion (Proposition 2) -- tests whether P1's null in Models A-C",
        "  masks a real effect concentrated in stress periods, averaged against a",
        "  near-zero effect in calm periods.",
    ]
    (config.OUTPUT / "table_p1_stress_dependent.txt").write_text("\n".join(e_lines), encoding="utf-8")
    pd.DataFrame(e_rows).to_csv(config.OUTPUT / "table_p1_stress_dependent.csv", index=False, float_format="%.6f")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    def _plot_panel(ax, specs, title):
        for results, color, label in specs:
            hs, coefs, ci_lo, ci_hi = [], [], [], []
            for h in HORIZONS:
                r = results[h]["res"]
                if r is None:
                    continue
                hs.append(h)
                coefs.append(r.params[INTERACT_VAR])
                ci = r.conf_int().loc[INTERACT_VAR]
                ci_lo.append(ci["lower"])
                ci_hi.append(ci["upper"])
            if hs:
                ax.plot(hs, coefs, color=color, lw=2, marker="o", markersize=5, label=label)
                ax.fill_between(hs, ci_lo, ci_hi, color=color, alpha=0.15)
        ax.axhline(0, color=GRAY, lw=1, ls="--")
        ax.set_xlabel("Horizon h (quarters)")
        ax.set_ylabel("delta_h: coefficient on (shock x linkage)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_xticks(HORIZONS)

    _plot_panel(axes[0],
        [(results_A, NAVY, "A: bank+quarter FE (primary)"),
         (results_B, BLUE, "B: bank FE only (robustness)")],
        "Identification comparison\n(contemporaneous L_it)")
    _plot_panel(axes[1],
        [(results_A, NAVY, "A: contemporaneous L_it"),
         (results_C, RED, "C: lagged L_{i,t-1}"),
         (results_D, "#006633", "D: bank-mean L (\"type\")")],
        "Linkage-variable comparison\n(bank+quarter FE throughout)")
    fig.suptitle("P1: Does linkage intensity alter monetary policy transmission?", y=1.03)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_p1_impulse_response.png", bbox_inches="tight")
    plt.close()

    hs_e, calm_vals, stress_diff_vals, total_vals, total_ci_lo, total_ci_hi = [], [], [], [], [], []
    for h in HORIZONS:
        res = results_E[h]["res"]
        if res is None:
            continue
        hs_e.append(h)
        calm_vals.append(res.params[INTERACT_VAR])
        stress_diff_vals.append(res.params["shock_x_L_x_stress"])
        total = res.params[INTERACT_VAR] + res.params["shock_x_L_x_stress"]
        total_vals.append(total)
        try:
            cov_matrix = res.cov
            var_total = (cov_matrix.loc[INTERACT_VAR, INTERACT_VAR]
                        + cov_matrix.loc["shock_x_L_x_stress", "shock_x_L_x_stress"]
                        + 2 * cov_matrix.loc[INTERACT_VAR, "shock_x_L_x_stress"])
            se_total = np.sqrt(var_total) if var_total > 0 else np.nan
        except Exception:
            se_total = np.nan
        total_ci_lo.append(total - 1.96 * se_total)
        total_ci_hi.append(total + 1.96 * se_total)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(hs_e, calm_vals, color=BLUE, lw=2, marker="o", markersize=5, label="Calm quarters (baseline)")
    ax.plot(hs_e, total_vals, color=RED, lw=2, marker="s", markersize=5, label="High-stress quarters (total effect)")
    ax.fill_between(hs_e, total_ci_lo, total_ci_hi, color=RED, alpha=0.15)
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel("Coefficient on (mp_shock x L_it)")
    ax.set_title("Does transmission heterogeneity depend on financial stress?\n"
                 "(NFCI top-tercile quarters vs. calm quarters)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(HORIZONS)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_p1_stress_dependent.png", bbox_inches="tight")
    plt.close()

    # Model F: linkage-type tercile group comparison #

    groups = ["Low", "Medium", "High"]
    results_F = {g: {} for g in groups}
    for h in HORIZONS:
        for g in groups:
            res, n_g, banks_g, dropped_g = _fit_model_F(df, h, g)
            results_F[g][h] = {"res": res, "n": n_g, "banks": banks_g}
            if dropped_g > 0:
                log.append(f"  h={h} group={g}: dropped {dropped_g} singleton obs before Model F")
            if res is not None and getattr(res, "_cov_note", None):
                log.append(f"  h={h} group={g}: {res._cov_note}")

    f_rows = []
    f_lines = [
        "TABLE: P1 Extension -- Linkage-Type Tercile Group Comparison",
        f"Groups by L_it_bankmean: Low<={q33:.4f}, Medium<={q67:.4f}, High>{q67:.4f}",
        f"Bank counts: {group_bank_counts.to_dict()}",
        "Bank FE only (no quarter FE -- need mp_shock's own coefficient, not just an",
        "interaction), estimated SEPARATELY within each group's subsample.", "",
        f"{'h':<4}{'Low':>16}{'Medium':>16}{'High':>16}{'High-Low diff':>16}{'p-value':>10}", "-" * 78,
    ]
    for h in HORIZONS:
        row = f"{h:<4}"
        coefs = {}
        for g in groups:
            r = results_F[g][h]["res"]
            if r is not None:
                c = r.params["mp_shock"]
                p = r.pvalues["mp_shock"]
                coefs[g] = (c, r.std_errors["mp_shock"])
                row += f"{f'{c:+.4f}{_stars(p)}':>16}"
            else:
                row += f"{'failed':>16}"
        if "High" in coefs and "Low" in coefs:
            diff = coefs["High"][0] - coefs["Low"][0]
            # Non-overlapping, independent bank subsamples: variances simply add, no covariance term needed.
            diff_se = np.sqrt(coefs["High"][1] ** 2 + coefs["Low"][1] ** 2)
            diff_t = diff / diff_se if diff_se > 0 else np.nan
            diff_p = 2 * (1 - stats.norm.cdf(np.abs(diff_t))) if not np.isnan(diff_t) else np.nan
            row += f"{f'{diff:+.4f}{_stars(diff_p)}':>16}{diff_p:>10.4f}"
            f_rows.append({"h": h, "low": coefs.get("Low", (np.nan,))[0],
                          "medium": coefs.get("Medium", (np.nan,))[0],
                          "high": coefs.get("High", (np.nan,))[0],
                          "high_minus_low": diff, "high_minus_low_se": diff_se, "high_minus_low_pval": diff_p})
        else:
            row += f"{'n/a':>16}{'n/a':>10}"
        f_lines.append(row)
    f_lines += [
        "-" * 78, "",
        "Notes: beta_h = coefficient on mp_shock, estimated separately within each",
        "  linkage-type tercile's subsample (bank FE only, two-way clustered SE).",
        "'High-Low diff' = High-group beta_h minus Low-group beta_h; independent",
        "  (non-overlapping) subsamples, so its SE is simply sqrt(se_High^2 + se_Low^2).",
        "*** p<0.01  ** p<0.05  * p<0.10 (two-sided, standard normal).",
        "Complements Model D (bank-mean L_it as an interaction): here the comparison is",
        "  between separately-estimated coefficients across sorted groups, the standard",
        "  approach in the bank-lending-channel literature (Kashyap-Stein 1995).",
    ]
    (config.OUTPUT / "table_p1_linkage_groups.txt").write_text("\n".join(f_lines), encoding="utf-8")
    pd.DataFrame(f_rows).to_csv(config.OUTPUT / "table_p1_linkage_groups.csv", index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    group_colors = {"Low": GREEN, "Medium": BLUE, "High": RED}
    for g in groups:
        hs_g, coefs_g, ci_lo_g, ci_hi_g = [], [], [], []
        for h in HORIZONS:
            r = results_F[g][h]["res"]
            if r is None:
                continue
            hs_g.append(h)
            coefs_g.append(r.params["mp_shock"])
            ci = r.conf_int().loc["mp_shock"]
            ci_lo_g.append(ci["lower"])
            ci_hi_g.append(ci["upper"])
        if hs_g:
            ax.plot(hs_g, coefs_g, color=group_colors[g], lw=2, marker="o", markersize=5,
                    label=f"{g} linkage-type")
            ax.fill_between(hs_g, ci_lo_g, ci_hi_g, color=group_colors[g], alpha=0.12)
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel("beta_h: coefficient on mp_shock")
    ax.set_title("Are highly-interconnected banks fundamentally different?\n"
                 "Transmission effect estimated separately by linkage-type tercile")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(HORIZONS)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_p1_linkage_groups.png", bbox_inches="tight")
    plt.close()

    (config.OUTPUT / "p1_regression_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()