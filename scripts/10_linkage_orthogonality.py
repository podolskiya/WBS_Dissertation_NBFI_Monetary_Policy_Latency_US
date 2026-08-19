"""
10_linkage_orthogonality.py
-----------------------------
Supervisor feedback: how much of L_it is just a repackaged version of
standard, already-observable bank characteristics (size, capital ratio,
loan growth, etc.), and how much is genuinely orthogonal to them?

Two complementary answers, deliberately kept simple -- this is a
descriptive diagnostic, not a hypothesis test, so no bootstrap or
clustered inference here (unlike 02/04/05/06):

  1. Pairwise correlations, both bank-level (cross-section, using each
     bank's full-sample average) and panel-level (pooled bank-quarter
     observations). These can differ -- a bank-level correlation asks
     "are persistently high-linkage banks a certain type of bank?";
     panel-level asks "does linkage move with these characteristics
     quarter to quarter?"

  2. Bank-level regression of mean L_it on the full observable set,
     reporting R^2. (1 - R^2) is the sharper answer to "how much is
     orthogonal" -- a handful of pairwise correlations can't rule out
     L_it being a linear combination of several observables even if no
     single correlation looks strong on its own.

Characteristics used: log_assets (size), capital_ratio, ci_loan_growth,
asset_growth, has_sec_lending. credit_growth_4q is deliberately
excluded -- it's the P1/P2 outcome variable, not a bank characteristic,
and including it here would blur that distinction.

INPUT   data/clean/panel_clean.parquet
OUTPUT  outputs/table_linkage_orthogonality.txt
        outputs/table_linkage_orthogonality.csv
        outputs/fig_linkage_orthogonality_heatmap.png
        outputs/linkage_orthogonality_log.txt
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
config = import_module("00_config")

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
NAVY, BLUE, RED, GRAY = "#003366", "#0066cc", "#cc0000", "#666666"

CHARACTERISTICS = ["log_assets", "capital_ratio", "ci_loan_growth",
                   "asset_growth", "has_sec_lending"]
CHAR_LABELS = {
    "log_assets": "Log assets (size)", "capital_ratio": "Capital ratio",
    "ci_loan_growth": "C&I loan growth", "asset_growth": "Asset growth",
    "has_sec_lending": "Has sec. lending",
}


def main():
    print("\n" + "=" * 65)
    print("10_linkage_orthogonality.py  --  L_it vs. observable bank characteristics")
    print("=" * 65)

    log = ["LINKAGE ORTHOGONALITY DIAGNOSTIC", "=" * 65, ""]

    # [1] Load
    print("\n[1] Loading panel_clean.parquet")
    df = pd.read_parquet(config.F_PANEL_CLEAN)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])
    print(f"  {len(df):,} rows, {df['rssd9001'].nunique()} banks")

    all_vars = ["L_it"] + CHARACTERISTICS

    # [2] Panel-level (pooled) correlations
    print("\n[2] Panel-level correlations (pooled bank-quarter observations)")
    panel_corr = df[all_vars].corr()
    log.append("PANEL-LEVEL (pooled bank-quarter) correlations with L_it:")
    for var in CHARACTERISTICS:
        c = panel_corr.loc["L_it", var]
        print(f"  L_it vs {CHAR_LABELS[var]:<20} {c:+.4f}")
        log.append(f"  L_it vs {var:<16} {c:+.4f}")

    # [3] Bank-level (cross-sectional) correlations
    print("\n[3] Bank-level correlations (cross-section, full-sample bank means)")
    bank_means = df.groupby("rssd9001")[all_vars].mean()
    bank_corr = bank_means.corr()
    log.append("\nBANK-LEVEL (cross-sectional mean) correlations with L_it:")
    for var in CHARACTERISTICS:
        c = bank_corr.loc["L_it", var]
        print(f"  L_it vs {CHAR_LABELS[var]:<20} {c:+.4f}")
        log.append(f"  L_it vs {var:<16} {c:+.4f}")

    # [4] Bank-level R^2: how much of L_it's cross-sectional variation
    # do these observables jointly explain?
    print("\n[4] Bank-level regression: mean L_it on all observables (plain OLS, no inference)")
    reg_data = bank_means.dropna()
    y = reg_data["L_it"].values
    X = np.column_stack([np.ones(len(reg_data)), reg_data[CHARACTERISTICS].values])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f"  N banks: {len(reg_data)}")
    print(f"  R^2 = {r2:.4f}  (share of cross-bank L_it variation explained by "
          f"{', '.join(CHARACTERISTICS)})")
    print(f"  1 - R^2 = {1 - r2:.4f}  (share orthogonal to these observables)")
    log.append(f"\nBank-level OLS: mean L_it ~ {' + '.join(CHARACTERISTICS)}")
    log.append(f"  N banks: {len(reg_data)}")
    log.append(f"  R^2 = {r2:.4f}")
    log.append(f"  1 - R^2 = {1 - r2:.4f}  <- share of L_it's cross-bank variation")
    log.append(f"  NOT explained by these standard observables (*1)")
    coef_lines = ["  Coefficients (plain OLS, not for inference -- see note *1):",
                  f"    const           {beta[0]:+.4f}"]
    for i, var in enumerate(CHARACTERISTICS):
        coef_lines.append(f"    {var:<16}{beta[i+1]:+.4f}")
    for line in coef_lines:
        print(line)
        log.append(line)

    # [5] Save correlation matrix and heatmap
    print("\n[5] Saving correlation matrix and heatmap")
    panel_corr.to_csv(config.OUTPUT / "table_linkage_orthogonality.csv", float_format="%.4f")

    txt_lines = [
        "TABLE: L_it vs. Observable Bank Characteristics",
        "Answers: how much of L_it is captured by standard, already-observable",
        "bank characteristics, vs. genuinely orthogonal to them?",
        "=" * 60, "",
        f"{'Characteristic':<20}{'Panel-level':>14}{'Bank-level':>14}", "-" * 48,
    ]
    for var in CHARACTERISTICS:
        txt_lines.append(f"{CHAR_LABELS[var]:<20}{panel_corr.loc['L_it', var]:>14.4f}"
                         f"{bank_corr.loc['L_it', var]:>14.4f}")
    txt_lines += [
        "-" * 48, "",
        f"Bank-level R^2 (mean L_it ~ all characteristics jointly): {r2:.4f}",
        f"  -> {1 - r2:.4f} of cross-bank L_it variation is NOT explained by",
        "     size, capital ratio, loan growth, asset growth, or sec. lending",
        "     activity jointly -- i.e. genuinely orthogonal to these standard",
        "     observables, not just weakly correlated with each individually.",
        "",
        "Notes: Panel-level = pooled bank-quarter observations (does linkage",
        "  move with these characteristics quarter to quarter?). Bank-level =",
        "  cross-section of each bank's full-sample mean (are persistently",
        "  high-linkage banks a particular type of bank?). (*1)",
    ]
    txt = "\n".join(txt_lines)
    (config.OUTPUT / "table_linkage_orthogonality.txt").write_text(txt, encoding="utf-8")
    print(f"  Saved -> {config.OUTPUT / 'table_linkage_orthogonality.txt'}")
    print("\n" + txt)

    fig, ax = plt.subplots(figsize=(7, 6))
    corr_display = panel_corr.loc[["L_it"] + CHARACTERISTICS, ["L_it"] + CHARACTERISTICS]
    im = ax.imshow(corr_display.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["L_it"] + [CHAR_LABELS[v] for v in CHARACTERISTICS]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr_display.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr_display.values[i, j]) > 0.5 else "black", fontsize=9)
    ax.set_title("L_it vs. observable bank characteristics\n(panel-level, pooled bank-quarter correlations)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_linkage_orthogonality_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {config.OUTPUT / 'fig_linkage_orthogonality_heatmap.png'}")

    log_text = "\n".join(log)
    (config.OUTPUT / "linkage_orthogonality_log.txt").write_text(log_text, encoding="utf-8")
    print(f"\n  Log -> {config.OUTPUT / 'linkage_orthogonality_log.txt'}")


if __name__ == "__main__":
    main()

# ============================================================
# APPENDIX
# ============================================================
# *1  This is a plain OLS / correlation diagnostic, not a hypothesis
#     test -- no clustered SEs or bootstrap, unlike the P1/P2 models.
#     The point here is descriptive (how much shared variation exists),
#     not inferential (is a specific coefficient significantly nonzero).
#     If a formal test of any individual relationship is later needed,
#     it should use the same clustered-bootstrap approach as 05/06, not
#     the plain R^2 reported here.