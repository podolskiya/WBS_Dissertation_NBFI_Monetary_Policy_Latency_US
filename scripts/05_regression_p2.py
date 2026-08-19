"""
05_regression_p2.py
--------------------
PURPOSE
    Estimate the P2 Growth-at-Risk (GaR) quantile regression:
    does bank-NBFI linkage intensity (L_it) predict downside tail
    risk in four-quarter-ahead C&I credit growth, incremental to
    conventional prudential indicators?

    Implements three falsifiable tests from the dissertation plan:
      H0(FRAGILITY)   |beta_L(tau=0.10)| = |beta_L(tau=0.50)| -- no tail amplification
      H0(REDUNDANCY)  beta_L(tau=0.10) = 0 given capital_ratio -- L_it adds nothing
                       beyond conventional prudential metrics at the left tail
      H0(ASYMMETRY)   beta_L(tau=0.10) = beta_L(tau=0.90) -- symmetric effect
                       (fragility) vs asymmetric (pure volatility)

SPECIFICATION
    Outcome: credit_growth_4q = log(CI_{i,t+4} / CI_{i,t}), built via
             calendar-based self-merge in 02_linkage_index.py.

    Model A (baseline, no L_it):
        Q_tau(credit_growth_4q) = a + b1*nfci + b2*capital_ratio
                                  + b3*log_assets + b4*ci_loan_growth
    Model B (main, with L_it):
        Q_tau(credit_growth_4q) = a + b1*nfci + b2*L_it + b3*capital_ratio
                                  + b4*log_assets + b5*ci_loan_growth

    Quantiles: tau in {0.10, 0.25, 0.50, 0.75, 0.90}

INFERENCE
    Pairs cluster bootstrap (B=500), clustering on rssd9001 (bank).
    Banks resampled with replacement; within-bank time series kept
    intact in each replication -- corrects for within-bank
    autocorrelation without parametric assumptions on the error
    structure. SEs = empirical SD of the bootstrap distribution;
    95% CIs = percentile method; p-values two-sided vs standard normal.

INPUT   data/clean/panel_macro.parquet

OUTPUT  outputs/table_p2_quantreg.csv          (all quantiles, both models)
        outputs/table_p2_quantreg.txt          (formatted, write-up ready)
        outputs/table_p2_nested_comparison.txt (Model A vs B pseudo-R2, H0_REDUNDANCY)
        outputs/fig_p2_coef_by_quantile.png
        outputs/p2_regression_log.txt

REFERENCES
    Adrian, T., Boyarchenko, N. and Giannone, D. (2019) 'Vulnerable
    Growth', American Economic Review, 109(4), pp. 1263-1289.
"""

import sys
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.regression.quantile_regression import QuantReg
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

QUANTILES   = [0.10, 0.25, 0.50, 0.75, 0.90]
N_BOOTSTRAP = 500
SEED        = 42
OUTCOME     = "credit_growth_4q"

REGRESSORS_A = ["nfci", "capital_ratio", "log_assets", "ci_loan_growth"]
REGRESSORS_B = ["nfci", "L_it", "capital_ratio", "log_assets", "ci_loan_growth"]

VAR_LABELS = {
    "const": "Constant", "nfci": "NFCI (fin. conditions)", "L_it": "L_it (linkage index)",
    "capital_ratio": "Capital ratio", "log_assets": "Log total assets",
    "ci_loan_growth": "C&I loan growth (QoQ)",
}


def _add_const(X: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X.values])


def _fit_quantreg(y, X, tau):
    try:
        return QuantReg(y, X).fit(q=tau, max_iter=2000, p_tol=1e-6)
    except Exception:
        return None


def _pseudo_r2(y, X, tau, params):
    """Koenker-Machado pseudo-R2 at quantile tau."""
    def _rho(u, t):
        return u * (t - (u < 0).astype(float))
    resid_full = y - X @ params
    resid_null = y - np.quantile(y, tau)
    loss_full  = np.sum(_rho(resid_full, tau))
    loss_null  = np.sum(_rho(resid_null, tau))
    return np.nan if loss_null == 0 else 1.0 - loss_full / loss_null


def _bootstrap_ses(y, X, bank_ids, tau, n_boot, rng):
    """Pairs cluster bootstrap: resample banks with replacement, keep
    each bank's full time series intact within a replication."""
    unique_banks = np.unique(bank_ids)
    n_banks = len(unique_banks)
    boot_coefs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_banks, size=n_banks, replace=True)
        idx = np.concatenate([np.where(bank_ids == b)[0] for b in sampled])
        result = _fit_quantreg(y[idx], X[idx], tau)
        if result is not None:
            boot_coefs.append(result.params)
    if not boot_coefs:
        n_p = X.shape[1]
        return (np.full((1, n_p), np.nan), np.full(n_p, np.nan), np.full(n_p, np.nan), 0)
    boot_coefs = np.array(boot_coefs)
    return (boot_coefs, np.percentile(boot_coefs, 2.5, axis=0),
            np.percentile(boot_coefs, 97.5, axis=0), len(boot_coefs))


def _stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def main():
    t0 = time.time()
    print("\n" + "=" * 65)
    print("05_regression_p2.py  --  P2 Growth-at-Risk quantile regression")
    print("=" * 65)

    log = [
        "P2 QUANTILE REGRESSION AUDIT LOG", "=" * 65,
        f"Outcome: {OUTCOME}", f"Quantiles: {QUANTILES}",
        f"Bootstrap: B={N_BOOTSTRAP}, seed={SEED}, cluster=rssd9001 (bank)", "",
    ]

    # ── [1] Load and prepare estimation sample ──────────────────────────
    print("\n[1] Loading panel and preparing estimation sample")
    df = pd.read_parquet(config.F_PANEL_MACRO)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])

    all_vars = list(dict.fromkeys(REGRESSORS_B + [OUTCOME, "rssd9001", "year"]))
    est = df[all_vars].dropna().copy().reset_index(drop=True)

    n_total, n_est = len(df), len(est)
    n_banks = est["rssd9001"].nunique()
    print(f"  Full panel:        {n_total:,} rows, {df['rssd9001'].nunique()} banks")
    print(f"  Estimation sample: {n_est:,} rows, {n_banks} banks")
    print(f"  Rows dropped:      {n_total - n_est:,}")
    print(f"  Year range:        {est['year'].min()} - {est['year'].max()}")
    log += [f"Estimation sample: {n_est:,} rows | {n_banks} banks",
            f"Rows dropped from full panel ({n_total:,}): {n_total - n_est:,}", ""]

    y = est[OUTCOME].values
    bank_ids = est["rssd9001"].values
    X_A = _add_const(est[REGRESSORS_A])
    X_B = _add_const(est[REGRESSORS_B])
    colnames_A = ["const"] + REGRESSORS_A
    colnames_B = ["const"] + REGRESSORS_B
    rng = np.random.default_rng(SEED)

    # ── [2] Full-sample point estimates ─────────────────────────────────
    print("\n[2] Full-sample point estimates (Model A and Model B)")
    results = {"A": {}, "B": {}}
    for tau in QUANTILES:
        fit_A = _fit_quantreg(y, X_A, tau)
        fit_B = _fit_quantreg(y, X_B, tau)
        if fit_A is None or fit_B is None:
            print(f"  WARNING: full-sample fit failed at tau={tau}")
            continue
        results["A"][tau] = {"params": fit_A.params, "pseudo_r2": _pseudo_r2(y, X_A, tau, fit_A.params)}
        results["B"][tau] = {"params": fit_B.params, "pseudo_r2": _pseudo_r2(y, X_B, tau, fit_B.params)}
        print(f"  tau={tau:.2f}  beta_L_it={fit_B.params[colnames_B.index('L_it')]:+.4f}  "
              f"pseudo-R2(A)={results['A'][tau]['pseudo_r2']:.4f}  "
              f"pseudo-R2(B)={results['B'][tau]['pseudo_r2']:.4f}")

    # ── [3] Bootstrap inference ──────────────────────────────────────────
    print(f"\n[3] Bootstrap inference (B={N_BOOTSTRAP}, clustering on bank)")
    print("    Progress: ", end="", flush=True)
    boot_results = {"A": {}, "B": {}}
    for tau in QUANTILES:
        for model_name, X, colnames in [("A", X_A, colnames_A), ("B", X_B, colnames_B)]:
            boot_coefs, ci_lo, ci_hi, n_ok = _bootstrap_ses(y, X, bank_ids, tau, N_BOOTSTRAP, rng)
            se = np.std(boot_coefs, axis=0, ddof=1)
            tstat = results[model_name][tau]["params"] / se
            pval = 2 * (1 - stats.norm.cdf(np.abs(tstat)))
            boot_results[model_name][tau] = {"se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
                                              "tstat": tstat, "pval": pval, "n_ok": n_ok}
            if n_ok < 0.95 * N_BOOTSTRAP:
                log.append(f"  WARNING: tau={tau} model={model_name} bootstrap success = "
                           f"{n_ok}/{N_BOOTSTRAP}")
        print(f"tau={tau:.2f}({boot_results['B'][tau]['n_ok']}ok) ", end="", flush=True)
    print()

    # ── [4] Build results tables ─────────────────────────────────────────
    print("\n[4] Building results tables")
    rows_csv = []
    for model_name, colnames in [("B", colnames_B), ("A", colnames_A)]:
        for var_idx, var in enumerate(colnames):
            for tau in QUANTILES:
                coef = results[model_name][tau]["params"][var_idx]
                b = boot_results[model_name][tau]
                rows_csv.append({
                    "model": model_name, "variable": var, "tau": tau, "coef": coef,
                    "se": b["se"][var_idx], "tstat": b["tstat"][var_idx], "pval": b["pval"][var_idx],
                    "ci_lo": b["ci_lo"][var_idx], "ci_hi": b["ci_hi"][var_idx],
                    "stars": _stars(b["pval"][var_idx]),
                    "pseudo_r2": results[model_name][tau]["pseudo_r2"],
                })
    df_csv = pd.DataFrame(rows_csv)
    df_csv.to_csv(config.OUTPUT / "table_p2_quantreg.csv", index=False, float_format="%.6f")
    print(f"  CSV saved -> {config.OUTPUT / 'table_p2_quantreg.csv'}")

    # ── [5] Formatted text table (Model B) ──────────────────────────────
    print("\n[5] Writing formatted text table")
    sep_wide = "-" * 90
    hdr_tau = "".join(f"  {f'tau={t:.2f}':>14}" for t in QUANTILES)
    txt_lines = [
        "TABLE: P2 Growth-at-Risk Quantile Regression (Model B)",
        "Outcome: credit_growth_4q = log(CI_{i,t+4} / CI_{i,t})",
        f"N = {n_est:,} bank-quarters | {n_banks} banks",
        f"Bootstrap: B={N_BOOTSTRAP} replications, pairs cluster on bank (seed={SEED})",
        "", f"{'Variable':<24}" + hdr_tau, sep_wide,
    ]
    for var_idx, var in enumerate(colnames_B):
        label = VAR_LABELS.get(var, var)
        coef_row, se_row, ci_row = f"  {label:<22}", f"  {'':22}", f"  {'':22}"
        for tau in QUANTILES:
            coef = results["B"][tau]["params"][var_idx]
            b = boot_results["B"][tau]
            se, pval = b["se"][var_idx], b["pval"][var_idx]
            ci_lo, ci_hi = b["ci_lo"][var_idx], b["ci_hi"][var_idx]
            coef_row += f"  {coef:>10.4f}{_stars(pval):<3}"
            se_row   += f"  {'(' + f'{se:.4f}' + ')':>13}"
            ci_row   += f"  {'[' + f'{ci_lo:.3f},{ci_hi:.3f}' + ']':>13}"
        txt_lines += [coef_row, se_row, ci_row, ""]
    txt_lines += [sep_wide, ""]
    pr2_row = f"  {'Pseudo-R2':<22}"
    for tau in QUANTILES:
        pr2_row += f"  {results['B'][tau]['pseudo_r2']:>13.4f}"
    txt_lines += [pr2_row, "", "Notes: Dependent variable is four-quarter cumulative log C&I loan growth.",
        "Bootstrap standard errors in parentheses (B=500 pairs cluster bootstrap, "
        "clustering on bank).", "95% percentile confidence intervals in brackets.",
        "*** p<0.01  ** p<0.05  * p<0.10 (two-sided, standard normal).",
        "L_it = (rcfd3819 + rcfd3433 + rcfdj458 + rconj454) / rcfd2170 (raw ratio, not standardised).",
        "All regressors measured at time t; outcome at t+4.",
    ]
    txt = "\n".join(txt_lines)
    (config.OUTPUT / "table_p2_quantreg.txt").write_text(txt, encoding="utf-8")
    print(f"  TXT saved -> {config.OUTPUT / 'table_p2_quantreg.txt'}")
    print("\n" + txt)

    # ── [6] Nested model comparison (H0_REDUNDANCY) ─────────────────────
    print("\n[6] Nested model comparison: Model A vs Model B")
    nested_lines = [
        "TABLE: Nested Model Comparison (H0_REDUNDANCY test)",
        "Does L_it add predictive power for downside credit risk beyond",
        "conventional prudential indicators?", "",
        f"{'Quantile':<10}  {'PsR2(A)':>10}  {'PsR2(B)':>10}  {'Delta':>10}"
        f"  {'beta_L_it':>10}  {'SE':>8}  {'p-value':>8}  {'':>4}", "-" * 78,
    ]
    lit_idx = colnames_B.index("L_it")
    for tau in QUANTILES:
        pr2_A, pr2_B = results["A"][tau]["pseudo_r2"], results["B"][tau]["pseudo_r2"]
        delta = pr2_B - pr2_A
        coef_L = results["B"][tau]["params"][lit_idx]
        se_L   = boot_results["B"][tau]["se"][lit_idx]
        pval_L = boot_results["B"][tau]["pval"][lit_idx]
        nested_lines.append(
            f"  tau={tau:.2f}    {pr2_A:>10.4f}  {pr2_B:>10.4f}  {delta:>10.4f}  "
            f"{coef_L:>10.4f}  {se_L:>8.4f}  {pval_L:>8.4f}  {_stars(pval_L):<4}"
        )
    nested_lines += [
        "-" * 78, "", "Notes: Pseudo-R2 is the Koenker-Machado goodness-of-fit statistic.",
        "Model A: nfci + capital_ratio + log_assets + ci_loan_growth (no L_it).",
        "Model B: Model A + L_it.",
        "A positive Delta Pseudo-R2 at tau=0.10 with significant beta_L_it rejects "
        "H0(REDUNDANCY).",
    ]
    nested_txt = "\n".join(nested_lines)
    (config.OUTPUT / "table_p2_nested_comparison.txt").write_text(nested_txt, encoding="utf-8")
    print(f"  Saved -> {config.OUTPUT / 'table_p2_nested_comparison.txt'}")
    print("\n" + nested_txt)

    # ── [7] Hypothesis test summary ──────────────────────────────────────
    print("\n[7] Hypothesis test summary")
    b010 = results["B"][0.10]["params"][lit_idx]
    b050 = results["B"][0.50]["params"][lit_idx]
    b090 = results["B"][0.90]["params"][lit_idx]
    se010 = boot_results["B"][0.10]["se"][lit_idx]
    pval010 = boot_results["B"][0.10]["pval"][lit_idx]
    pval090 = boot_results["B"][0.90]["pval"][lit_idx]
    sd_L = df["L_it"].std()

    frag = "REJECT (tail amplification present)" if abs(b010) > abs(b050) else "FAIL TO REJECT"
    asym = "REJECT (asymmetric/fragility)" if b010 < b090 else "FAIL TO REJECT"
    redu = "REJECT" if pval010 < 0.10 else "FAIL TO REJECT"

    summary_lines = [
        "", "HYPOTHESIS TEST RESULTS", "-" * 65,
        f"beta_L_it(tau=0.10) = {b010:+.6f}  SE={se010:.6f}  p={pval010:.4f}{_stars(pval010)}",
        f"beta_L_it(tau=0.50) = {b050:+.6f}",
        f"beta_L_it(tau=0.90) = {b090:+.6f}", "",
        f"H0(FRAGILITY):  |beta(0.10)|={abs(b010):.4f} vs |beta(0.50)|={abs(b050):.4f}  -> {frag}",
        f"H0(ASYMMETRY):  beta(0.10)={b010:+.4f} vs beta(0.90)={b090:+.4f}  -> {asym}",
        f"H0(REDUNDANCY): beta_L_it(tau=0.10) p={pval010:.4f}  -> {redu} at 10% level", "",
        f"1SD increase in L_it ({sd_L:.4f} units):",
        f"  tau=0.10: {b010 * sd_L:+.4f} log pts",
        f"  tau=0.50: {b050 * sd_L:+.4f} log pts",
        f"  tau=0.90: {b090 * sd_L:+.4f} log pts",
    ]
    for line in summary_lines:
        print(line)
    log += summary_lines

    # ── [8] Coefficient plot ─────────────────────────────────────────────
    print("\n[8] Generating coefficient plot")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    tau_arr = np.array(QUANTILES)
    coefs_B = np.array([results["B"][t]["params"][lit_idx] for t in QUANTILES])
    ci_lo_B = np.array([boot_results["B"][t]["ci_lo"][lit_idx] for t in QUANTILES])
    ci_hi_B = np.array([boot_results["B"][t]["ci_hi"][lit_idx] for t in QUANTILES])

    axes[0].fill_between(tau_arr, ci_lo_B, ci_hi_B, color=BLUE, alpha=0.18, label="95% CI")
    axes[0].plot(tau_arr, coefs_B, color=NAVY, lw=2.5, marker="o", markersize=6, label="beta_L_it")
    axes[0].axhline(0, color=GRAY, lw=1, ls="--")
    axes[0].scatter([0.10], [b010], color=RED, zorder=5, s=80)
    axes[0].annotate(f"tau=0.10\n{b010:+.4f}{_stars(pval010)}", xy=(0.10, b010),
                     xytext=(0.15, b010 - abs(b010) * 0.5 - 0.02), fontsize=9, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    axes[0].set_xlabel("Quantile (tau)"); axes[0].set_ylabel("Coefficient on L_it")
    axes[0].set_title("beta_L_it across quantiles")
    axes[0].set_xticks(QUANTILES); axes[0].legend(frameon=False, fontsize=9)

    pr2_A_vals = [results["A"][t]["pseudo_r2"] for t in QUANTILES]
    pr2_B_vals = [results["B"][t]["pseudo_r2"] for t in QUANTILES]
    x = np.arange(len(QUANTILES)); w = 0.35
    axes[1].bar(x - w/2, pr2_A_vals, width=w, color=GRAY, alpha=0.8, label="Model A (no L_it)")
    axes[1].bar(x + w/2, pr2_B_vals, width=w, color=NAVY, alpha=0.85, label="Model B (+ L_it)")
    axes[1].set_xlabel("Quantile (tau)"); axes[1].set_ylabel("Pseudo-R2")
    axes[1].set_title("Model fit: A vs B"); axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{t:.2f}" for t in QUANTILES]); axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("P2 Growth-at-Risk: L_it and downside C&I credit tail risk", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_p2_coef_by_quantile.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {config.OUTPUT / 'fig_p2_coef_by_quantile.png'}")

    # ── Save log ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log += ["", f"Runtime: {elapsed:.1f} seconds"]
    (config.OUTPUT / "p2_regression_log.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\n  Log -> {config.OUTPUT / 'p2_regression_log.txt'}")
    print(f"\n  Done. Runtime: {elapsed:.1f}s")
    print("  Next: python 06_robustness.py")


if __name__ == "__main__":
    main()