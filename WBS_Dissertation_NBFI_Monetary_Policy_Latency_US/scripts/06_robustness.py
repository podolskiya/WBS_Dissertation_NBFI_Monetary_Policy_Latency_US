"""
06_robustness.py

Four pre-specified robustness checks: P2 with lagged linkage, P2 by L_it component, P2 large-bank split, P1 ZLB/COVID exclusion. See
README.md and the dissertation Methodology section for rationale. Reuses 04_regression_p1.py / 05_regression_p2.py as imported modules
so results can't silently diverge from the validated baseline specs.

INPUT   data/clean/panel_macro.parquet
OUTPUT  outputs/table_p2_lagged_L.txt / .csv
        outputs/table_p2_component_decomposition.txt / .csv
        outputs/table_p2_large_bank_split.txt / .csv
        outputs/table_p1_zlb_excluded.txt / .csv
        outputs/fig_p2_component_decomposition.png
        outputs/fig_p1_zlb_robustness.png
        outputs/06_robustness_log.txt
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
config = import_module("00_config")
p1 = import_module("04_regression_p1")
p2 = import_module("05_regression_p2")
 
warnings.filterwarnings("ignore")
 
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
NAVY, BLUE, RED, GREEN, GRAY = "#003366", "#0066cc", "#cc0000", "#006633", "#666666"
N_LARGE_BANKS = 10
 
 
def _stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""
 
 
def _run_p2_variant(est_df, l_col, label, log):
    """Full Model-B-style GaR quantile regression (bootstrap inference
    included) with l_col substituted for L_it. Reuses 05_regression_p2's
    tested _fit_quantreg / _bootstrap_ses / _pseudo_r2 directly."""
    regressors = ["nfci", l_col, "capital_ratio", "log_assets", "ci_loan_growth"]
    all_vars = list(dict.fromkeys(regressors + [p2.OUTCOME, "rssd9001"]))
    est = est_df[all_vars].dropna().copy().reset_index(drop=True)
    n_est, n_banks = len(est), est["rssd9001"].nunique()
    log.append(f"  [{label}] N={n_est:,} rows, {n_banks} banks")
    y = est[p2.OUTCOME].values
    bank_ids = est["rssd9001"].values
    X = p2._add_const(est[regressors])
    colnames = ["const"] + regressors
    l_idx = colnames.index(l_col)
    rng = np.random.default_rng(p2.SEED)
    results = {}
    print(f"  [{label}] ", end="", flush=True)
    for tau in p2.QUANTILES:
        fit = p2._fit_quantreg(y, X, tau)
        if fit is None:
            print(f"tau={tau}:failed ", end="", flush=True)
            continue
        pr2 = p2._pseudo_r2(y, X, tau, fit.params)
        boot_coefs, ci_lo, ci_hi, n_ok = p2._bootstrap_ses(y, X, bank_ids, tau, p2.N_BOOTSTRAP, rng)
        se = np.std(boot_coefs, axis=0, ddof=1)
        tstat = fit.params / se
        pval = 2 * (1 - stats.norm.cdf(np.abs(tstat)))
        results[tau] = {
            "coef": fit.params[l_idx], "se": se[l_idx], "pval": pval[l_idx],
            "ci_lo": ci_lo[l_idx], "ci_hi": ci_hi[l_idx], "pseudo_r2": pr2, "n_ok": n_ok,
        }
        print(f"tau={tau:.2f}({n_ok}ok) ", end="", flush=True)
    print()
    return results, n_est, n_banks
 
 
def main():
    t0 = time.time()
    print("Runs several full bootstrap quantile regressions; ~30-45 min.")
    log = ["06 ROBUSTNESS CHECKS AUDIT LOG", "=" * 65, ""]
 
    df = pd.read_parquet(config.F_PANEL_MACRO)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])
    df = p1._build_alt_linkage_vars(df)  # adds L_it_lag1, L_it_bankmean
 
    # P2: lagged linkage (L_it_lag1)#
    log.append("\n[1] P2 LAGGED LINKAGE ROBUSTNESS")
    res_lag, n_lag, banks_lag = _run_p2_variant(df, "L_it_lag1", "Lagged L (t-1)", log)
    rows = [{"tau": tau, **{k: v for k, v in r.items() if k != "n_ok"}, "stars": _stars(r["pval"])}
            for tau, r in res_lag.items()]
    pd.DataFrame(rows).to_csv(config.OUTPUT / "table_p2_lagged_L.csv", index=False, float_format="%.6f")
    lines = [
        "TABLE: P2 Robustness -- Lagged Linkage (L_{i,t-1})",
        "Tests whether the contemporaneous-L_it tail-risk result survives",
        "using the pre-outcome-window linkage value (avoids simultaneity,",
        "though P2 already has a natural lead-lag structure).", "",
        f"N = {n_lag:,} bank-quarters | {banks_lag} banks", "",
        f"{'Quantile':<10}{'coef':>12}{'SE':>10}{'p-value':>10}{'':>4}", "-" * 46,
    ]
    for tau, r in res_lag.items():
        lines.append(f"tau={tau:.2f}   {r['coef']:>10.4f}{r['se']:>10.4f}{r['pval']:>10.4f}  {_stars(r['pval'])}")
    (config.OUTPUT / "table_p2_lagged_L.txt").write_text("\n".join(lines), encoding="utf-8")
 
    # P2: component decomposition #
    log.append("\n[2] P2 COMPONENT DECOMPOSITION")
    components = {
        "L_middle": "SLCs + securities lending (both continuous components)",
        "L_guarantees": "Financial standby LCs only",
        "L_securities": "Securities lending only (GFC-era channel)",
        "L_lines": "Credit lines to FIs only (2010+, private-credit-like channel)",
    }
    comp_results = {}
    for col, desc in components.items():
        res, n_c, banks_c = _run_p2_variant(df, col, col, log)
        comp_results[col] = res
    comp_rows = []
    for col, res in comp_results.items():
        for tau, r in res.items():
            comp_rows.append({"component": col, "tau": tau, "coef": r["coef"], "se": r["se"],
                              "pval": r["pval"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                              "stars": _stars(r["pval"])})
    pd.DataFrame(comp_rows).to_csv(config.OUTPUT / "table_p2_component_decomposition.csv",
                                     index=False, float_format="%.6f")
    lines = [
        "TABLE: P2 Robustness -- Component Decomposition",
        "Which part of L_it drives the tail-risk (tau=0.10) result?", "",
        f"{'Component':<16}" + "".join(f"{'tau='+f'{t:.2f}':>16}" for t in p2.QUANTILES), "-" * 96,
    ]
    for col, desc in components.items():
        row = f"{col:<16}"
        for tau in p2.QUANTILES:
            r = comp_results[col].get(tau)
            if r:
                cell = f"{r['coef']:+.4f}{_stars(r['pval'])}"
            else:
                cell = "n/a"
            row += f"{cell:>16}"
        lines.append(row)
    lines += ["", "Component definitions:"]
    for col, desc in components.items():
        lines.append(f"  {col:<14} {desc}")
    lines += ["", "Compare to L_it (main index, from table_p2_quantreg.txt):",
              "  the tau=0.10 coefficient there combines all three components."]
    (config.OUTPUT / "table_p2_component_decomposition.txt").write_text("\n".join(lines), encoding="utf-8")
 
    fig, ax = plt.subplots(figsize=(8, 5))
    comp_labels = list(components.keys())
    coefs_010 = [comp_results[c][0.10]["coef"] if 0.10 in comp_results[c] else np.nan for c in comp_labels]
    ci_lo_010 = [comp_results[c][0.10]["ci_lo"] if 0.10 in comp_results[c] else np.nan for c in comp_labels]
    ci_hi_010 = [comp_results[c][0.10]["ci_hi"] if 0.10 in comp_results[c] else np.nan for c in comp_labels]
    errs = [[c - lo for c, lo in zip(coefs_010, ci_lo_010)], [hi - c for c, hi in zip(coefs_010, ci_hi_010)]]
    colors_bar = [NAVY, BLUE, RED, GREEN]
    ax.bar(range(len(comp_labels)), coefs_010, color=colors_bar, alpha=0.85, yerr=errs, capsize=4)
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    ax.set_xticks(range(len(comp_labels)))
    ax.set_xticklabels(comp_labels, rotation=20, ha="right")
    ax.set_ylabel("Coefficient at tau=0.10 (95% CI)")
    ax.set_title("Which L_it component drives the tail-risk (tau=0.10) result?")
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_p2_component_decomposition.png", bbox_inches="tight")
    plt.close()
 
    # P2: large-bank split #
    log.append(f"\n[3] P2 LARGE-BANK SPLIT (top {N_LARGE_BANKS} banks by mean rcfd2170)")
    bank_mean_assets = df.groupby("rssd9001")["rcfd2170"].mean()
    large_banks = set(bank_mean_assets.nlargest(N_LARGE_BANKS).index)
    df_excl = df[~df["rssd9001"].isin(large_banks)].copy()
    df_only = df[df["rssd9001"].isin(large_banks)].copy()
    res_excl, n_excl, banks_excl = _run_p2_variant(df_excl, "L_it", "Large-bank EXCLUDED", log)
    res_only, n_only, banks_only = _run_p2_variant(df_only, "L_it", "Large-bank ONLY", log)
    if banks_only < 10:
        log.append(f"  CAUTION: large-bank-only subsample has only {banks_only} bank clusters -- "
                   f"bootstrap SEs here are illustrative, not reliable inference.")
    split_rows = []
    for label, res in [("full_sample_see_table_p2_quantreg", None), ("excluded", res_excl), ("only", res_only)]:
        if res is None:
            continue
        for tau, r in res.items():
            split_rows.append({"subsample": label, "tau": tau, "coef": r["coef"], "se": r["se"],
                               "pval": r["pval"], "stars": _stars(r["pval"])})
    pd.DataFrame(split_rows).to_csv(config.OUTPUT / "table_p2_large_bank_split.csv",
                                      index=False, float_format="%.6f")
    lines = [
        f"TABLE: P2 Robustness -- Large-Bank Split (top {N_LARGE_BANKS} banks by mean assets)",
        "Does the tail-risk result survive excluding the banks that dominate",
        "aggregate NBFI-linkage dollar exposure (see 02's concentration diagnostic)?", "",
        f"Excluding top {N_LARGE_BANKS}: N={n_excl:,}, {banks_excl} banks",
        f"Only top {N_LARGE_BANKS}:      N={n_only:,}, {banks_only} banks", "",
        f"{'Quantile':<10}{'Excluded coef':>16}{'  p':>8}{'Only coef':>16}{'  p':>8}", "-" * 58,
    ]
    for tau in p2.QUANTILES:
        re_ = res_excl.get(tau)
        ro_ = res_only.get(tau)
        se_str = f"{re_['coef']:+.4f}{_stars(re_['pval'])}" if re_ else "n/a"
        so_str = f"{ro_['coef']:+.4f}{_stars(ro_['pval'])}" if ro_ else "n/a"
        pe = f"{re_['pval']:.3f}" if re_ else "n/a"
        po = f"{ro_['pval']:.3f}" if ro_ else "n/a"
        lines.append(f"tau={tau:.2f}   {se_str:>16}{pe:>8}{so_str:>16}{po:>8}")
    if banks_only < 10:
        lines += ["", f"CAUTION: large-bank-only subsample has only {banks_only} bank clusters --",
                 "bootstrap SEs here are illustrative, not reliable inference."]
    (config.OUTPUT / "table_p2_large_bank_split.txt").write_text("\n".join(lines), encoding="utf-8")
 
    # P1: ZLB/COVID exclusion #
    log.append("\n[4] P1 ZLB/COVID EXCLUSION ROBUSTNESS")
    df_p1 = p1._build_horizon_outcomes(df.copy())
    n_zlb = (df_p1["fedfunds"] <= 0.25).sum()
    log.append(f"  ZLB rows excluded: {n_zlb:,} of {len(df_p1):,}")
    df_p1_excl = df_p1[df_p1["fedfunds"] > 0.25].copy()
    results_full, results_zlbex = {}, {}
    for h in p1.HORIZONS:
        res_full, n_full, banks_full, _ = p1._fit_model_A(df_p1, h)
        res_ex, n_ex, banks_ex, _ = p1._fit_model_A(df_p1_excl, h)
        results_full[h] = res_full
        results_zlbex[h] = res_ex
    zlb_rows = []
    for h in p1.HORIZONS:
        rf, re_ = results_full[h], results_zlbex[h]
        zlb_rows.append({
            "h": h,
            "coef_full": rf.params[p1.INTERACT_VAR] if rf is not None else np.nan,
            "pval_full": rf.pvalues[p1.INTERACT_VAR] if rf is not None else np.nan,
            "coef_zlb_excluded": re_.params[p1.INTERACT_VAR] if re_ is not None else np.nan,
            "pval_zlb_excluded": re_.pvalues[p1.INTERACT_VAR] if re_ is not None else np.nan,
        })
    pd.DataFrame(zlb_rows).to_csv(config.OUTPUT / "table_p1_zlb_excluded.csv", index=False, float_format="%.6f")
    lines = [
        "TABLE: P1 Robustness -- ZLB/COVID Exclusion",
        "Model A (primary spec), excluding quarters at the effective zero",
        "lower bound (fedfunds <= 0.25%), where conventional rate shocks",
        "are close to meaningless.", "",
        f"{'h':<4}{'Full sample':>16}{'  p':>8}{'ZLB-excluded':>16}{'  p':>8}", "-" * 54,
    ]
    for row in zlb_rows:
        cf, cz = row["coef_full"], row["coef_zlb_excluded"]
        pf, pz = row["pval_full"], row["pval_zlb_excluded"]
        sf = f"{cf:+.4f}{_stars(pf)}" if not np.isnan(cf) else "failed"
        sz = f"{cz:+.4f}{_stars(pz)}" if not np.isnan(cz) else "failed"
        pf_str = f"{pf:.3f}" if not np.isnan(pf) else "n/a"
        pz_str = f"{pz:.3f}" if not np.isnan(pz) else "n/a"
        lines.append(f"{row['h']:<4}{sf:>16}{pf_str:>8}{sz:>16}{pz_str:>8}")
    (config.OUTPUT / "table_p1_zlb_excluded.txt").write_text("\n".join(lines), encoding="utf-8")
 
    fig, ax = plt.subplots(figsize=(8, 5.5))
    hs = p1.HORIZONS
    coefs_full = [results_full[h].params[p1.INTERACT_VAR] if results_full[h] is not None else np.nan for h in hs]
    coefs_ex = [results_zlbex[h].params[p1.INTERACT_VAR] if results_zlbex[h] is not None else np.nan for h in hs]
    ax.plot(hs, coefs_full, color=NAVY, lw=2, marker="o", markersize=5, label="Full sample")
    ax.plot(hs, coefs_ex, color=RED, lw=2, marker="s", markersize=5, label="ZLB/COVID excluded")
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel("delta_h: coefficient on (mp_shock x L_it)")
    ax.set_title("P1 Robustness: does the result depend on ZLB-period quarters?")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(hs)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_p1_zlb_robustness.png", bbox_inches="tight")
    plt.close()
 
    # P1: shock-rotation sensitivity #
    log.append("\n[5] P1 SHOCK-ROTATION SENSITIVITY (median vs poor man's rotation)")
    if "mp_shock_median" not in df_p1.columns:
        print("  [5] SKIPPED: mp_shock_median not found in panel -- check 03_macro_merge.py output")
        log.append("  SKIPPED: mp_shock_median column missing from panel_macro.parquet")
    else:
        # Swap the CONTENT of mp_shock rather than write a second fitting
        # path -- _fit_model_A always reads the shock off this column name
        # by design (see 04_regression_p1.py), so this guarantees the two
        # runs share every specification detail except the shock series.
        df_p1_median = df_p1.copy()
        df_p1_median["mp_shock"] = df_p1_median["mp_shock_median"]
 
        results_medrot = {}
        for h in p1.HORIZONS:
            res_med, n_med, banks_med, _ = p1._fit_model_A(df_p1_median, h)
            results_medrot[h] = res_med
 
        rotation_rows = []
        for h in p1.HORIZONS:
            rf, rm = results_full[h], results_medrot[h]
            rotation_rows.append({
                "h": h,
                "coef_poor_mans": rf.params[p1.INTERACT_VAR] if rf is not None else np.nan,
                "pval_poor_mans": rf.pvalues[p1.INTERACT_VAR] if rf is not None else np.nan,
                "coef_median_rotation": rm.params[p1.INTERACT_VAR] if rm is not None else np.nan,
                "pval_median_rotation": rm.pvalues[p1.INTERACT_VAR] if rm is not None else np.nan,
            })
        pd.DataFrame(rotation_rows).to_csv(config.OUTPUT / "table_p1_shock_rotation.csv",
                                            index=False, float_format="%.6f")
 
        lines = [
            "TABLE: P1 Robustness -- Shock-Rotation Sensitivity",
            "Model A (primary spec), mp_shock (poor man's rotation, MP_pm)",
            "versus mp_shock_median (median rotation, MP_median). Same",
            "specification, same sample, only the shock series differs.", "",
            f"{'h':<4}{'Poor mans':>16}{'  p':>8}{'Median rot.':>16}{'  p':>8}", "-" * 54,
        ]
        agree_count, n_valid_h = 0, 0
        for row in rotation_rows:
            cf, cm = row["coef_poor_mans"], row["coef_median_rotation"]
            pf, pm = row["pval_poor_mans"], row["pval_median_rotation"]
            sf = f"{cf:+.4f}{_stars(pf)}" if not np.isnan(cf) else "failed"
            sm = f"{cm:+.4f}{_stars(pm)}" if not np.isnan(cm) else "failed"
            pf_str = f"{pf:.3f}" if not np.isnan(pf) else "n/a"
            pm_str = f"{pm:.3f}" if not np.isnan(pm) else "n/a"
            lines.append(f"{row['h']:<4}{sf:>16}{pf_str:>8}{sm:>16}{pm_str:>8}")
            if not np.isnan(cf) and not np.isnan(cm):
                n_valid_h += 1
                if np.sign(cf) == np.sign(cm):
                    agree_count += 1
        lines += ["", f"Sign agreement: {agree_count} of {n_valid_h} horizons "
                       f"(both series have valid estimates)."]
        (config.OUTPUT / "table_p1_shock_rotation.txt").write_text("\n".join(lines), encoding="utf-8")
        log.append(f"  Sign agreement across horizons: {agree_count}/{n_valid_h}")
 
        fig, ax = plt.subplots(figsize=(8, 5.5))
        coefs_pm = [results_full[h].params[p1.INTERACT_VAR] if results_full[h] is not None else np.nan for h in hs]
        coefs_med = [results_medrot[h].params[p1.INTERACT_VAR] if results_medrot[h] is not None else np.nan for h in hs]
        ax.plot(hs, coefs_pm, color=NAVY, lw=2, marker="o", markersize=5, label="Poor man's rotation (primary, MP_pm)")
        ax.plot(hs, coefs_med, color=RED, lw=2, marker="s", markersize=5, label="Median rotation (MP_median)")
        ax.axhline(0, color=GRAY, lw=1, ls="--")
        ax.set_xlabel("Horizon h (quarters)")
        ax.set_ylabel("delta_h: coefficient on (shock x L_it)")
        ax.set_title("P1 Robustness: does the result depend on the shock's sign-restriction rotation?")
        ax.legend(frameon=False, fontsize=9)
        ax.set_xticks(hs)
        fig.tight_layout()
        fig.savefig(config.OUTPUT / "fig_p1_shock_rotation.png", bbox_inches="tight")
        plt.close()
 
    elapsed = time.time() - t0
    log += ["", f"Runtime: {elapsed:.1f} seconds"]
    (config.OUTPUT / "06_robustness_log.txt").write_text("\n".join(log), encoding="utf-8")
 
 
if __name__ == "__main__":
    main()