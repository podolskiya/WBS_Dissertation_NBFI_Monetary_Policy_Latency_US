"""
06_robustness.py
-----------------
PURPOSE
    Four robustness checks agreed with supervisor / motivated by prior
    results, none of them run to chase significance -- all were
    specified before seeing their output:

    1. P2, LAGGED LINKAGE: re-run the GaR quantile regression with
       L_{i,t-1} instead of contemporaneous L_it. P2 already has a
       natural lead-lag structure (L_it at t predicts growth from t to
       t+4), so simultaneity is a weaker concern than it was for P1,
       but worth checking directly.
    2. P2, COMPONENT DECOMPOSITION: re-run the tau=0.10 (and all
       quantiles) regression separately for each L_it component --
       L_middle, L_guarantees, L_securities, L_lines -- to identify
       which piece of the index drives the tail-risk result. Directly
       relevant to the supervisor's re-pitch comment: if the credit-
       lines-to-FIs component (2010+, the private-credit-like channel)
       drives the effect rather than securities lending (GFC-era,
       unrelated to private credit), that is real evidence the index
       is capturing something private-credit-like despite RCFD1520
       being unavailable.
    3. P2, LARGE-BANK SPLIT: re-run Model B excluding, and then using
       only, the top 10 banks by mean total assets (the same banks
       identified in 02_linkage_index.py's concentration diagnostic as
       holding ~93% of aggregate NBFI-linkage dollar exposure). Tests
       whether the tail-risk finding is a genuine cross-sectional
       pattern or an artefact of a handful of very large banks.
    4. P1, ZLB/COVID EXCLUSION: re-run Model A (primary local
       projections spec) excluding quarters at the effective zero
       lower bound (fedfunds <= 0.25%), where conventional rate
       shocks are close to meaningless and mostly reflect forward
       guidance / QE rather than the shock series used here.

INPUT   data/clean/panel_macro.parquet

OUTPUT  outputs/table_p2_lagged_L.txt / .csv
        outputs/table_p2_component_decomposition.txt / .csv
        outputs/table_p2_large_bank_split.txt / .csv
        outputs/table_p1_zlb_excluded.txt / .csv
        outputs/fig_p2_component_decomposition.png
        outputs/fig_p1_zlb_robustness.png
        outputs/06_robustness_log.txt

NOTE: reuses the tested fitting logic from 04_regression_p1.py and
05_regression_p2.py directly (imported as modules) rather than
re-implementing it, so nothing here can silently diverge from the
already-validated baseline specifications.
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

N_LARGE_BANKS = 10  # matches the "top 10 banks hold ~93%" framing in 02's concentration diagnostic


def _stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _run_p2_variant(est_df, l_col, label, log):
    """
    Run the full Model-B-style GaR quantile regression (bootstrap
    inference included) with l_col substituted for L_it. Reuses
    05_regression_p2's tested _fit_quantreg / _bootstrap_ses /
    _pseudo_r2 directly.
    """
    regressors = ["nfci", l_col, "capital_ratio", "log_assets", "ci_loan_growth"]
    all_vars = list(dict.fromkeys(regressors + [p2.OUTCOME, "rssd9001"]))
    est = est_df[all_vars].dropna().copy().reset_index(drop=True)
    n_est, n_banks = len(est), est["rssd9001"].nunique()
    print(f"  [{label}] N={n_est:,} rows, {n_banks} banks")
    log.append(f"  [{label}] N={n_est:,} rows, {n_banks} banks")

    y = est[p2.OUTCOME].values
    bank_ids = est["rssd9001"].values
    X = p2._add_const(est[regressors])
    colnames = ["const"] + regressors
    l_idx = colnames.index(l_col)
    rng = np.random.default_rng(p2.SEED)

    results = {}
    for tau in p2.QUANTILES:
        fit = p2._fit_quantreg(y, X, tau)
        if fit is None:
            print(f"    tau={tau}: fit failed")
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
        print(f"    tau={tau:.2f}: beta={fit.params[l_idx]:+.4f}{_stars(pval[l_idx])} "
              f"(SE={se[l_idx]:.4f}, {n_ok}/{p2.N_BOOTSTRAP} boot ok)")
    return results, n_est, n_banks


def main():
    t0 = time.time()
    print("\n" + "=" * 65)
    print("06_robustness.py  --  P1/P2 robustness checks")
    print("=" * 65)
    print("\nWARNING: this runs several full bootstrap quantile regressions")
    print("and will take roughly 30-45 minutes. Let it run to completion.\n")

    log = ["06 ROBUSTNESS CHECKS AUDIT LOG", "=" * 65, ""]

    # ── [0] Load panel, build alt linkage vars ──────────────────────────
    print("[0] Loading panel and building variant linkage variables")
    df = pd.read_parquet(config.F_PANEL_MACRO)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])
    df = p1._build_alt_linkage_vars(df)  # adds L_it_lag1, L_it_bankmean
    print(f"  Panel: {len(df):,} rows, {df['rssd9001'].nunique()} banks")

    # ══════════════════════════════════════════════════════════════════
    # [1] P2 ROBUSTNESS: lagged linkage (L_{i,t-1})
    # ══════════════════════════════════════════════════════════════════
    print("\n[1] P2 robustness: lagged linkage (L_it_lag1)")
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
    print(f"  Saved -> {config.OUTPUT / 'table_p2_lagged_L.txt'}")

    # ══════════════════════════════════════════════════════════════════
    # [2] P2 ROBUSTNESS: component decomposition
    # ══════════════════════════════════════════════════════════════════
    print("\n[2] P2 robustness: component decomposition")
    log.append("\n[2] P2 COMPONENT DECOMPOSITION")
    components = {
        "L_middle":     "SLCs + securities lending (both continuous components)",
        "L_guarantees": "Financial standby LCs only",
        "L_securities": "Securities lending only (GFC-era channel)",
        "L_lines":      "Credit lines to FIs only (2010+, private-credit-like channel)",
    }
    comp_results = {}
    for col, desc in components.items():
        print(f"\n  Component: {col} ({desc})")
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
    print(f"  Saved -> {config.OUTPUT / 'table_p2_component_decomposition.txt'}")

    # Figure: component coefficients at tau=0.10 across components
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
    print(f"  Saved -> {config.OUTPUT / 'fig_p2_component_decomposition.png'}")

    # ══════════════════════════════════════════════════════════════════
    # [3] P2 ROBUSTNESS: large-bank split
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[3] P2 robustness: large-bank split (top {N_LARGE_BANKS} banks by mean assets)")
    log.append(f"\n[3] P2 LARGE-BANK SPLIT (top {N_LARGE_BANKS} banks by mean rcfd2170)")
    bank_mean_assets = df.groupby("rssd9001")["rcfd2170"].mean()
    large_banks = set(bank_mean_assets.nlargest(N_LARGE_BANKS).index)
    print(f"  Identified {len(large_banks)} large banks (by mean total assets)")

    df_excl = df[~df["rssd9001"].isin(large_banks)].copy()
    df_only = df[df["rssd9001"].isin(large_banks)].copy()

    print(f"\n  -- Excluding top {N_LARGE_BANKS} banks --")
    res_excl, n_excl, banks_excl = _run_p2_variant(df_excl, "L_it", "Large-bank EXCLUDED", log)
    print(f"\n  -- Only top {N_LARGE_BANKS} banks --")
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
    print(f"  Saved -> {config.OUTPUT / 'table_p2_large_bank_split.txt'}")

    # ══════════════════════════════════════════════════════════════════
    # [4] P1 ROBUSTNESS: ZLB/COVID exclusion
    # ══════════════════════════════════════════════════════════════════
    print("\n[4] P1 robustness: ZLB exclusion (fedfunds <= 0.25%)")
    log.append("\n[4] P1 ZLB/COVID EXCLUSION ROBUSTNESS")
    df_p1 = p1._build_horizon_outcomes(df.copy())
    n_zlb = (df_p1["fedfunds"] <= 0.25).sum()
    print(f"  Quarters at ZLB (fedfunds<=0.25%): {n_zlb:,} of {len(df_p1):,} rows")
    log.append(f"  ZLB rows excluded: {n_zlb:,} of {len(df_p1):,}")
    df_p1_excl = df_p1[df_p1["fedfunds"] > 0.25].copy()

    results_full, results_zlbex = {}, {}
    for h in p1.HORIZONS:
        res_full, n_full, banks_full, _ = p1._fit_model_A(df_p1, h)
        res_ex, n_ex, banks_ex, _ = p1._fit_model_A(df_p1_excl, h)
        results_full[h] = res_full
        results_zlbex[h] = res_ex
        c_full = res_full.params[p1.INTERACT_VAR] if res_full is not None else np.nan
        c_ex = res_ex.params[p1.INTERACT_VAR] if res_ex is not None else np.nan
        print(f"  h={h}: full sample delta_h={c_full:+.4f} (N={n_full:,})  |  "
              f"ZLB-excluded delta_h={c_ex:+.4f} (N={n_ex:,})")

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
    print(f"  Saved -> {config.OUTPUT / 'table_p1_zlb_excluded.txt'}")

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
    print(f"  Saved -> {config.OUTPUT / 'fig_p1_zlb_robustness.png'}")

    # ── Save log ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log += ["", f"Runtime: {elapsed:.1f} seconds"]
    (config.OUTPUT / "06_robustness_log.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\n  Log -> {config.OUTPUT / '06_robustness_log.txt'}")
    print(f"\n  Done. Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()