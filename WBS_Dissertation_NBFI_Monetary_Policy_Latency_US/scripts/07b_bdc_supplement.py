"""
07_bdc_supplement.py

Supplementary, descriptive BDC (Business Development Company) triangulation exhibit, complementing the
bank-side L_it story with independent data.

OUTPUT  outputs/table_bdc_crosscheck.txt
        outputs/table_bdc_leverage_trend.csv
        outputs/table_bdc_concentration.txt
        outputs/fig_bdc_leverage_trend.png
        outputs/fig_bdc_concentration_lorenz.png
        outputs/fig_bdc_vs_bank_linkage.png
        outputs/bdc_supplement_log.txt
"""
import sys
import re
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

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
NAVY, BLUE, RED, GREEN, GRAY = "#003366", "#0066cc", "#cc0000", "#006633", "#666666"
FINANCIAL_COLS = ["total_assets", "total_debt", "nav", "revolving_credit",
                  "bank_debt", "undrawn_credit", "variable_rate_debt", "cash_and_equiv"]


def _parse_capiq_quarter(q: str):
    """'FQ12010' -> 2010-03-31. Raw export uses 'FQ', not 'CQ' -- an
    earlier version of this script assumed the latter and silently
    produced all-NaT dates."""
    m = re.match(r"FQ(\d)(\d{4})", str(q))
    if not m:
        return pd.NaT
    quarter, year = int(m.group(1)), int(m.group(2))
    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.QuarterEnd(0)


def _parse_sec_period(p: str):
    """'CY2021Q4I' -> 2021-12-31."""
    m = re.match(r"CY(\d{4})Q(\d)I?", str(p))
    if not m:
        return pd.NaT
    year, quarter = int(m.group(1)), int(m.group(2))
    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.QuarterEnd(0)


def _load_capiq(path):
    df = pd.read_excel(path, sheet_name="Main")
    df = df[df["ticker"] != "-"].copy()
    df["pure_ticker"] = df["ticker"].str.split(":").str[-1].str.replace(".X", "", regex=False)
    df["quarter_end"] = df["quarter"].apply(_parse_capiq_quarter)
    for col in FINANCIAL_COLS:
        df[col] = df[col].replace(0, np.nan)
    return df


def _load_sec_xbrl(path):
    df = pd.read_csv(path)
    df["quarter_end"] = df["Period"].apply(_parse_sec_period)
    return df


def main():
    log = ["BDC SUPPLEMENT AUDIT LOG", "=" * 65, ""]

    capiq = _load_capiq(config.F_BDC_CAPIQ)
    n_tickers = capiq["pure_ticker"].nunique()
    log.append(f"Capital IQ: {len(capiq):,} rows, {n_tickers} tickers")

    # Zero-as-missing pattern check: pre-IPO prefix vs scattered gaps #

    scattered = []
    for tk, sub in capiq.groupby("pure_ticker"):
        sub = sub.sort_values("quarter_end")
        is_missing = sub["total_assets"].isna().values
        if is_missing.any():
            first_valid = is_missing.tolist().index(False) if False in is_missing else len(is_missing)
            if is_missing[first_valid:].any():
                scattered.append(tk)
    log.append(f"Tickers with scattered missing-data pattern (not explained by pre-IPO "
               f"timing alone): {len(scattered)} of {n_tickers} -- {scattered}")
    log.append("  Zero values treated as missing throughout (see docstring); this handles")
    log.append("  both pre-IPO gaps and scattered gaps consistently, at the cost of some")
    log.append("  loss of precision for affected ticker-quarters.")

    sec = _load_sec_xbrl(config.F_BDC_XBRL)
    log.append(f"\nSEC XBRL: {len(sec):,} rows, {sec['Pure_Ticker'].nunique()} tickers")

    cross_lines = ["TABLE: Capital IQ vs SEC XBRL Cross-Check", "=" * 60, ""]

    debt_tag = sec[sec["Tag"] == "DebtInstrumentCarryingAmount"][["Pure_Ticker", "quarter_end", "Value"]]
    debt_tag = debt_tag.rename(columns={"Value": "sec_total_debt"})
    merged_debt = capiq.merge(debt_tag, left_on=["pure_ticker", "quarter_end"],
                              right_on=["Pure_Ticker", "quarter_end"], how="inner")
    merged_debt = merged_debt.dropna(subset=["total_debt", "sec_total_debt"])
    if len(merged_debt) > 5:
        ratio = (merged_debt["sec_total_debt"] / merged_debt["total_debt"]).median()
        merged_debt["capiq_scaled"] = merged_debt["total_debt"] * ratio
        corr = merged_debt[["capiq_scaled", "sec_total_debt"]].corr().iloc[0, 1]
        cross_lines += [
            f"total_debt (CapIQ) vs DebtInstrumentCarryingAmount (SEC XBRL):",
            f"  N overlapping ticker-quarters: {len(merged_debt)}",
            f"  Inferred scale factor (SEC/CapIQ): {ratio:,.0f}",
            f"  Correlation after scaling: {corr:.4f}", "",
        ]
        log.append(f"total_debt cross-check: N={len(merged_debt)}, scale={ratio:,.0f}, corr={corr:.4f}")
    else:
        cross_lines.append(f"total_debt cross-check: insufficient overlap (N={len(merged_debt)}, need >5)")

    loc_tag = sec[sec["Tag"] == "LineOfCreditFacilityMaximumBorrowingCapacity"][["Pure_Ticker", "quarter_end", "Value"]]
    loc_tag = loc_tag.rename(columns={"Value": "sec_facility_capacity"})
    capiq["facility_capacity_proxy"] = capiq["revolving_credit"].fillna(0) + capiq["undrawn_credit"].fillna(0)
    merged_loc = capiq.merge(loc_tag, left_on=["pure_ticker", "quarter_end"],
                             right_on=["Pure_Ticker", "quarter_end"], how="inner")
    merged_loc = merged_loc[merged_loc["facility_capacity_proxy"] > 0].dropna(subset=["sec_facility_capacity"])
    if len(merged_loc) > 5:
        ratio2 = (merged_loc["sec_facility_capacity"] / merged_loc["facility_capacity_proxy"]).median()
        merged_loc["capiq_scaled"] = merged_loc["facility_capacity_proxy"] * ratio2
        corr2 = merged_loc[["capiq_scaled", "sec_facility_capacity"]].corr().iloc[0, 1]
        cross_lines += [
            "revolving_credit + undrawn_credit (CapIQ) vs LineOfCreditFacilityMaximumBorrowingCapacity (SEC XBRL):",
            f"  N overlapping ticker-quarters: {len(merged_loc)}",
            f"  Inferred scale factor (SEC/CapIQ): {ratio2:,.0f}",
            f"  Correlation after scaling: {corr2:.4f}",
        ]
        log.append(f"facility-capacity cross-check: N={len(merged_loc)}, scale={ratio2:,.0f}, corr={corr2:.4f}")
    else:
        cross_lines.append(f"facility-capacity cross-check: insufficient overlap (N={len(merged_loc)}, need >5)")

    (config.OUTPUT / "table_bdc_crosscheck.txt").write_text("\n".join(cross_lines), encoding="utf-8")

    # L1: sector-aggregate leverage trend #

    agg = (capiq.groupby("quarter_end")
             .agg(n_reporting=("total_assets", lambda x: x.notna().sum()),
                  sum_assets=("total_assets", "sum"),
                  sum_debt=("total_debt", "sum"),
                  sum_nav=("nav", "sum"),
                  sum_bank_debt=("bank_debt", "sum"),
                  sum_revolving=("revolving_credit", "sum"),
                  sum_undrawn=("undrawn_credit", "sum"))
             .reset_index())
    agg["leverage_debt_assets"] = agg["sum_debt"] / agg["sum_assets"]
    agg["leverage_debt_nav"] = agg["sum_debt"] / agg["sum_nav"]
    agg["bank_debt_share_of_total_debt"] = agg["sum_bank_debt"] / agg["sum_debt"]
    agg.to_csv(config.OUTPUT / "table_bdc_leverage_trend.csv", index=False, float_format="%.6f")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(agg["quarter_end"], agg["leverage_debt_assets"], color=NAVY, lw=2, label="Debt / Assets")
    axes[0].plot(agg["quarter_end"], agg["leverage_debt_nav"] / 10, color=BLUE, lw=1.5, ls="--",
                 label="Debt / NAV (÷10, for scale)")
    axes[0].set_xlabel("Quarter"); axes[0].set_ylabel("Ratio")
    axes[0].set_title("BDC sector aggregate leverage")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(agg["quarter_end"], agg["n_reporting"], color=GREEN, lw=2)
    axes[1].set_xlabel("Quarter"); axes[1].set_ylabel("Number of BDCs reporting")
    axes[1].set_title("BDC universe coverage over time\n(reflects IPO timing, not survivorship)")
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_bdc_leverage_trend.png", bbox_inches="tight")
    plt.close()

    # L2: BDC-side concentration diagnostic - mirrors 02 bank-side one #

    conc_lines = ["TABLE: BDC-Side Concentration Diagnostic",
                 "Mirrors the bank-side concentration check in 02_linkage_index.py --",
                 "is BDC-sector total assets dominated by a handful of large funds?",
                 "=" * 60, ""]
    shares = []
    for q, g in capiq.dropna(subset=["total_assets"]).groupby("quarter_end"):
        g = g.sort_values("total_assets", ascending=False)
        total = g["total_assets"].sum()
        if total <= 0:
            continue
        shares.append({
            "q": q, "n": len(g),
            "top5": g["total_assets"].head(5).sum() / total,
            "top10": g["total_assets"].head(10).sum() / total,
        })
    sh = pd.DataFrame(shares)
    conc_lines += [
        "Average share of aggregate BDC-sector total assets held by:",
        f"  Top 5 BDCs (of ~{sh.n.mean():.0f}):  {sh.top5.mean()*100:.1f}%  "
        f"(range {sh.top5.min()*100:.1f}-{sh.top5.max()*100:.1f}%)",
        f"  Top 10 BDCs:              {sh.top10.mean()*100:.1f}%  "
        f"(range {sh.top10.min()*100:.1f}-{sh.top10.max()*100:.1f}%)",
    ]
    (config.OUTPUT / "table_bdc_concentration.txt").write_text("\n".join(conc_lines), encoding="utf-8")

    bank_totals = capiq.dropna(subset=["total_assets"]).groupby("pure_ticker")["total_assets"].mean().sort_values()
    cum_n = np.arange(1, len(bank_totals) + 1) / len(bank_totals)
    cum_assets = (bank_totals.cumsum() / bank_totals.sum()).to_numpy(dtype=float)
    cum_n = np.asarray(cum_n, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="Perfect equality")
    ax.plot(cum_n, cum_assets, color=RED, lw=2, label="Lorenz curve (BDC total assets)")
    ax.fill_between(cum_n, cum_assets, cum_n, color=RED, alpha=0.1)
    ax.set_xlabel("Cumulative share of BDCs (smallest to largest)")
    ax.set_ylabel("Cumulative share of aggregate BDC-sector assets")
    ax.set_title("Concentration of BDC-sector total assets")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_bdc_concentration_lorenz.png", bbox_inches="tight")
    plt.close()

    # L3: BDC bank-debt vs bank-side L_lines triangulation #

    panel = pd.read_parquet(config.F_PANEL_MACRO)
    panel["rssd9999"] = pd.to_datetime(panel["rssd9999"])
    bank_side = panel.groupby("rssd9999")["L_lines"].mean().reset_index()
    bank_side = bank_side.rename(columns={"rssd9999": "quarter_end"})
    combined = agg.merge(bank_side, on="quarter_end", how="inner")
    combined.to_csv(config.OUTPUT / "table_bdc_vs_bank_linkage.csv", index=False, float_format="%.6f")

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(combined["quarter_end"], combined["sum_bank_debt"], color=NAVY, lw=2,
             label="BDC sector: aggregate bank_debt ($, CapIQ)")
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("BDC aggregate bank debt", color=NAVY)
    ax1.tick_params(axis="y", labelcolor=NAVY)
    ax2 = ax1.twinx()
    ax2.plot(combined["quarter_end"], combined["L_lines"], color=RED, lw=2, ls="--",
             label="Bank-side: mean L_lines (credit lines to FIs)")
    ax2.set_ylabel("Bank-side mean L_lines", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, fontsize=8.5)
    ax1.set_title("Triangulation: BDC-side bank borrowing vs bank-side credit-lines-to-FIs\n"
                 "(descriptive comparison, independent data sources -- not a merged regression)")
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_bdc_vs_bank_linkage.png", bbox_inches="tight")
    plt.close()

    if len(combined) > 5:
        corr_triangulate = combined[["sum_bank_debt", "L_lines"]].corr().iloc[0, 1]
        log.append(f"\nTriangulation correlation, LEVELS (BDC bank_debt, bank-side L_lines): {corr_triangulate:.4f}")
        # Both series likely trend upward together from general private-credit sector growth - a raw levels correlation is a classic spurious/correlation risk. QoQ change nets out any shared trend. #
        combined_sorted = combined.sort_values("quarter_end").copy()
        combined_sorted["d_bank_debt"] = combined_sorted["sum_bank_debt"].diff()
        combined_sorted["d_L_lines"] = combined_sorted["L_lines"].diff()
        diffed = combined_sorted.dropna(subset=["d_bank_debt", "d_L_lines"])
        if len(diffed) > 5:
            corr_diff = diffed[["d_bank_debt", "d_L_lines"]].corr().iloc[0, 1]
            log.append(f"Triangulation correlation, QoQ CHANGES (detrended): {corr_diff:.4f}")
            log.append("  (Levels correlation can be spurious if both series simply trend upward")
            log.append("  together over the sample; the QoQ-change correlation is the more")
            log.append("  defensible number to cite, since it nets out any shared trend.)")

    (config.OUTPUT / "bdc_supplement_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()