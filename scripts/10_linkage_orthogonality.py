"""
10_linkage_orthogonality.py

Diagnostic: how much of L_it is a repackaged version of standard, already-observable bank characteristics, and how much is genuinely
orthogonal to them? Descriptive only - no clustered SEs or bootstrap, unlike 02/04/05/06. S

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
CHARACTERISTICS = ["log_assets", "capital_ratio", "ci_loan_growth", "asset_growth"]
CHAR_LABELS = {
    "log_assets": "Log assets (size)", "capital_ratio": "Capital ratio",
    "ci_loan_growth": "C&I loan growth", "asset_growth": "Asset growth",
}
# Excluded from the characteristics set - it's a binarized version of one of L_it's own numerator components (See README) #

SEC_LENDING_VAR = "has_sec_lending"
SEC_LENDING_LABEL = "Has sec. lending"

def main():
    log = ["LINKAGE ORTHOGONALITY DIAGNOSTIC", "=" * 65, ""]

    df = pd.read_parquet(config.F_PANEL_CLEAN)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])
    all_vars = ["L_it"] + CHARACTERISTICS

    panel_corr = df[all_vars].corr()
    log.append("PANEL-LEVEL (pooled bank-quarter) correlations with L_it:")
    for var in CHARACTERISTICS:
        c = panel_corr.loc["L_it", var]
        log.append(f"  L_it vs {var:<16} {c:+.4f}")

    bank_means = df.groupby("rssd9001")[all_vars].mean()
    bank_corr = bank_means.corr()
    log.append("\nBANK-LEVEL (cross-sectional mean) correlations with L_it:")
    for var in CHARACTERISTICS:
        c = bank_corr.loc["L_it", var]
        log.append(f"  L_it vs {var:<16} {c:+.4f}")

    reg_data = bank_means.dropna()
    y = reg_data["L_it"].values
    X = np.column_stack([np.ones(len(reg_data)), reg_data[CHARACTERISTICS].values])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    log.append(f"\nBank-level OLS: mean L_it ~ {' + '.join(CHARACTERISTICS)}")
    log.append(f"  N banks: {len(reg_data)}")
    log.append(f"  R^2 = {r2:.4f}")
    log.append(f"  1 - R^2 = {1 - r2:.4f}  <- share of L_it's cross-bank variation")
    log.append(f"  NOT explained by these standard observables (descriptive only, see README)")
    coef_lines = ["  Coefficients (plain OLS, not for inference -- see README):",
                  f"    const           {beta[0]:+.4f}"]
    for i, var in enumerate(CHARACTERISTICS):
        coef_lines.append(f"    {var:<16}{beta[i+1]:+.4f}")
    log.extend(coef_lines)

    # Construction check, rather than folded into the R^2/correlation table above (see README) #

    sec_vars = ["L_it", SEC_LENDING_VAR]
    sec_panel_corr = df[sec_vars].corr().loc["L_it", SEC_LENDING_VAR]
    sec_bank_means = df.groupby("rssd9001")[sec_vars].mean()
    sec_bank_corr = sec_bank_means.corr().loc["L_it", SEC_LENDING_VAR]
    log.append(f"\n[CONSTRUCTION CHECK - not an independent characteristic, excluded from R^2 above]")
    log.append(f"L_it vs {SEC_LENDING_VAR} (= rcfd3433 > 0, one of L_it's own numerator terms):")
    log.append(f"  panel-level = {sec_panel_corr:+.4f}   bank-level = {sec_bank_corr:+.4f}")
    log.append(f"  Strong correlation expected by construction; confirms internal consistency")
    log.append(f"  of the L_it build, not orthogonality to independent observables.")

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
        f"Bank-level R^2 (mean L_it ~ size + capital ratio + loan growth + asset growth): {r2:.4f}",
        f"  -> {1 - r2:.4f} of cross-bank L_it variation is NOT explained by these four",
        "     conventional prudential/scale characteristics jointly - i.e. genuinely",
        "     orthogonal to them, not just weakly correlated with each individually.",
        "",
        "Notes: Panel-level = pooled bank-quarter observations (does linkage",
        "  move with these characteristics quarter to quarter?). Bank-level =",
        "  cross-section of each bank's full-sample mean (are persistently",
        "  high-linkage banks a particular type of bank?). Descriptive diagnostic,",
        "  not a hypothesis test.",
        "",
        f"[Construction check, excluded from the table/R^2 above -- see README]",
        f"{SEC_LENDING_LABEL:<20}{sec_panel_corr:>14.4f}{sec_bank_corr:>14.4f}",
    ]
    (config.OUTPUT / "table_linkage_orthogonality.txt").write_text("\n".join(txt_lines), encoding="utf-8")

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

    (config.OUTPUT / "linkage_orthogonality_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()