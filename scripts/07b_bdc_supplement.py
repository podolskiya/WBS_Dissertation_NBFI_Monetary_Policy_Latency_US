"""
07_bdc_supplement.py
---------------------
PURPOSE
    Supplementary, descriptive exhibit using Business Development
    Company (BDC) data -- the closest publicly-observable proxy for
    private credit specifically, complementing the bank-side L_it
    story (which measures bank exposure to NBFIs generally, not
    private credit funds specifically -- see 00_config.py notes on
    RCFD1520 being unavailable).

    This is deliberately a DESCRIPTIVE TRIANGULATION exhibit, not a
    merged regression: bank-BDC pair-level linking (which specific
    banks lend to which specific BDCs) is not attempted -- that would
    be its own research project. What this script does establish:
    does aggregate BDC sector leverage / bank-facility usage move in
    a way that is at least consistent with the bank-side linkage
    story, using genuinely independent data sources?

DATA SOURCES
    Primary: Capital IQ pull across the BDC industry classification,
             67 tickers resolved, quarterly, 2010Q1-2024Q3 (matches the
             main panel's window; see 00_config.py END_DATE note on the
             2024Q4 RC-L schedule retirement).
             Fields: total_assets, total_debt, nav, revolving_credit,
             bank_debt, undrawn_credit, variable_rate_debt, cash_and_equiv.
             Zero values are treated as missing (not genuine zero) --
             a real operating public BDC does not report literally $0
             total assets; zeros are a pre-IPO/reporting-gap artifact
             (confirmed: e.g. ARCC has zero such gaps since 2010, while
             BXSL/OBDC show a clean zero-prefix ending exactly at their
             IPO quarter -- but ~28 of 67 tickers show SCATTERED zeros
             not explained by IPO timing alone, likely fiscal-year
             misalignment or merger-related ticker transitions; treating
             zero as missing handles both cases consistently).

    Cross-check: SEC XBRL company-facts pull, 38 tickers, sparser/
             irregular period coverage (not a clean quarterly panel).
             Only LineOfCreditFacilityMaximumBorrowingCapacity is used
             as an independent check on Capital IQ's revolving_credit +
             undrawn_credit (both represent facility capacity, though
             not identically defined). DebtInstrumentCarryingAmount is
             used as a secondary check against Capital IQ's total_debt
             -- confirmed clean (zero duplicate ticker/period pairs, one
             consolidated value per company-period, not fragmented
             across individual debt instruments).

OUTPUT  outputs/table_bdc_crosscheck.txt        (CapIQ vs SEC XBRL agreement)
        outputs/table_bdc_leverage_trend.csv    (sector aggregate leverage, by quarter)
        outputs/table_bdc_concentration.txt     (BDC-side concentration diagnostic)
        outputs/fig_bdc_leverage_trend.png
        outputs/fig_bdc_concentration_lorenz.png
        outputs/fig_bdc_vs_bank_linkage.png     (BDC bank-debt vs bank-side L_lines)
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
    """'FQ12010' -> 2010-03-31 (fiscal quarter, treated as calendar-aligned
    -- confirmed against the actual raw export format, which uses 'FQ',
    not the 'CQ' notation an earlier version of this script assumed)."""
    m = re.match(r"FQ(\d)(\d{4})", str(q))
    if not m:
        return pd.NaT
    quarter, year = int(m.group(1)), int(m.group(2))
    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.QuarterEnd(0)


def _parse_sec_period(p: str):
    """'CY2021Q4I' -> 2021-12-31 (calendar quarter, quarter-end date)."""
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
    print("\n" + "=" * 65)
    print("07_bdc_supplement.py  --  BDC supplementary triangulation exhibit")
    print("=" * 65)

    log = ["BDC SUPPLEMENT AUDIT LOG", "=" * 65, ""]

    # ── [1] Load Capital IQ (primary) ────────────────────────────────────
    print("\n[1] Loading Capital IQ BDC panel")
    capiq = _load_capiq(config.F_BDC_CAPIQ)
    n_tickers = capiq["pure_ticker"].nunique()
    print(f"  {len(capiq):,} rows, {n_tickers} tickers, "
          f"{capiq['quarter_end'].min().date()} to {capiq['quarter_end'].max().date()}")
    log.append(f"Capital IQ: {len(capiq):,} rows, {n_tickers} tickers")

    # Data-quality check: zero-as-missing pattern (prefix vs scattered)
    scattered = []
    for tk, sub in capiq.groupby("pure_ticker"):
        sub = sub.sort_values("quarter_end")
        is_missing = sub["total_assets"].isna().values
        if is_missing.any():
            first_valid = is_missing.tolist().index(False) if False in is_missing else len(is_missing)
            if is_missing[first_valid:].any():
                scattered.append(tk)
    print(f"  Tickers with scattered (non-prefix) missing total_assets: {len(scattered)} of {n_tickers}")
    log.append(f"Tickers with scattered missing-data pattern (not explained by pre-IPO "
               f"timing alone): {len(scattered)} of {n_tickers} -- {scattered}")
    log.append("  Zero values treated as missing throughout (see docstring); this handles")
    log.append("  both pre-IPO gaps and scattered gaps consistently, at the cost of some")
    log.append("  loss of precision for affected ticker-quarters.")

    # ── [2] Load SEC XBRL cross-check ───────────────────────────────────
    print("\n[2] Loading SEC XBRL cross-check data")
    sec = _load_sec_xbrl(config.F_BDC_XBRL)
    print(f"  {len(sec):,} rows, {sec['Pure_Ticker'].nunique()} tickers")
    print(f"  Tags: {sec['Tag'].value_counts().to_dict()}")
    log.append(f"\nSEC XBRL: {len(sec):,} rows, {sec['Pure_Ticker'].nunique()} tickers")

    # ── [3] Cross-validation: CapIQ vs SEC XBRL ─────────────────────────
    print("\n[3] Cross-validating Capital IQ against SEC XBRL")
    cross_lines = ["TABLE: Capital IQ vs SEC XBRL Cross-Check", "=" * 60, ""]

    # 3a. total_debt vs DebtInstrumentCarryingAmount (both in raw $; SEC in $, CapIQ likely in $mm -- reconcile scale)
    debt_tag = sec[sec["Tag"] == "DebtInstrumentCarryingAmount"][["Pure_Ticker", "quarter_end", "Value"]]
    debt_tag = debt_tag.rename(columns={"Value": "sec_total_debt"})
    merged_debt = capiq.merge(debt_tag, left_on=["pure_ticker", "quarter_end"],
                              right_on=["Pure_Ticker", "quarter_end"], how="inner")
    merged_debt = merged_debt.dropna(subset=["total_debt", "sec_total_debt"])
    print(f"  Overlapping (ticker, quarter) for total_debt check: {len(merged_debt)}")
    if len(merged_debt) > 5:
        # infer scale factor (CapIQ often in $millions, SEC XBRL in raw $)
        ratio = (merged_debt["sec_total_debt"] / merged_debt["total_debt"]).median()
        print(f"  Median (SEC value / CapIQ value) ratio: {ratio:,.0f}  "
              f"(~1e6 implies CapIQ is in $ millions, SEC in raw $)")
        merged_debt["capiq_scaled"] = merged_debt["total_debt"] * ratio
        corr = merged_debt[["capiq_scaled", "sec_total_debt"]].corr().iloc[0, 1]
        print(f"  Correlation after scale adjustment: {corr:.4f}")
        cross_lines += [
            f"total_debt (CapIQ) vs DebtInstrumentCarryingAmount (SEC XBRL):",
            f"  N overlapping ticker-quarters: {len(merged_debt)}",
            f"  Inferred scale factor (SEC/CapIQ): {ratio:,.0f}",
            f"  Correlation after scaling: {corr:.4f}", "",
        ]
        log.append(f"total_debt cross-check: N={len(merged_debt)}, scale={ratio:,.0f}, corr={corr:.4f}")
    else:
        cross_lines.append("Insufficient overlap for total_debt cross-check.")
        print("  Insufficient overlap for cross-check")

    # 3b. revolving_credit + undrawn_credit vs LineOfCreditFacilityMaximumBorrowingCapacity
    loc_tag = sec[sec["Tag"] == "LineOfCreditFacilityMaximumBorrowingCapacity"][["Pure_Ticker", "quarter_end", "Value"]]
    loc_tag = loc_tag.rename(columns={"Value": "sec_facility_capacity"})
    capiq["facility_capacity_proxy"] = capiq["revolving_credit"].fillna(0) + capiq["undrawn_credit"].fillna(0)
    merged_loc = capiq.merge(loc_tag, left_on=["pure_ticker", "quarter_end"],
                             right_on=["Pure_Ticker", "quarter_end"], how="inner")
    merged_loc = merged_loc[merged_loc["facility_capacity_proxy"] > 0].dropna(subset=["sec_facility_capacity"])
    print(f"  Overlapping (ticker, quarter) for facility-capacity check: {len(merged_loc)}")
    if len(merged_loc) > 5:
        ratio2 = (merged_loc["sec_facility_capacity"] / merged_loc["facility_capacity_proxy"]).median()
        merged_loc["capiq_scaled"] = merged_loc["facility_capacity_proxy"] * ratio2
        corr2 = merged_loc[["capiq_scaled", "sec_facility_capacity"]].corr().iloc[0, 1]
        print(f"  Median scale ratio: {ratio2:,.0f}, correlation after scaling: {corr2:.4f}")
        cross_lines += [
            "revolving_credit + undrawn_credit (CapIQ) vs LineOfCreditFacilityMaximumBorrowingCapacity (SEC XBRL):",
            f"  N overlapping ticker-quarters: {len(merged_loc)}",
            f"  Inferred scale factor (SEC/CapIQ): {ratio2:,.0f}",
            f"  Correlation after scaling: {corr2:.4f}",
        ]
        log.append(f"facility-capacity cross-check: N={len(merged_loc)}, scale={ratio2:,.0f}, corr={corr2:.4f}")
    else:
        cross_lines.append("Insufficient overlap for facility-capacity cross-check "
                           "(SEC XBRL coverage of this tag is sparse -- see docstring).")
        print("  Insufficient overlap for cross-check (expected -- sparse SEC tag coverage)")

    cross_text = "\n".join(cross_lines)
    (config.OUTPUT / "table_bdc_crosscheck.txt").write_text(cross_text, encoding="utf-8")
    print(f"  Saved -> {config.OUTPUT / 'table_bdc_crosscheck.txt'}")

    # ══════════════════════════════════════════════════════════════════
    # LAYER 1: sector-aggregate leverage trend
    # ══════════════════════════════════════════════════════════════════
    print("\n[4] Layer 1: sector-aggregate leverage trend")
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
    print(f"  {len(agg)} quarters, {agg['n_reporting'].min()}-{agg['n_reporting'].max()} BDCs reporting per quarter")
    print(f"  Bank debt as share of total BDC debt: "
          f"{agg['bank_debt_share_of_total_debt'].iloc[0]:.1%} ({agg['quarter_end'].iloc[0].date()}) -> "
          f"{agg['bank_debt_share_of_total_debt'].iloc[-1]:.1%} ({agg['quarter_end'].iloc[-1].date()}), "
          f"min {agg['bank_debt_share_of_total_debt'].min():.1%}, max {agg['bank_debt_share_of_total_debt'].max():.1%}")
    print(f"  Saved -> {config.OUTPUT / 'table_bdc_leverage_trend.csv'}")

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
    print(f"  Saved -> {config.OUTPUT / 'fig_bdc_leverage_trend.png'}")

    # ══════════════════════════════════════════════════════════════════
    # LAYER 2: BDC-side concentration diagnostic (mirrors 02's bank-side one)
    # ══════════════════════════════════════════════════════════════════
    print("\n[5] Layer 2: BDC-side concentration diagnostic")
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
    conc_text = "\n".join(conc_lines)
    (config.OUTPUT / "table_bdc_concentration.txt").write_text(conc_text, encoding="utf-8")
    print(f"  Saved -> {config.OUTPUT / 'table_bdc_concentration.txt'}")
    print("\n" + conc_text)

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
    print(f"  Saved -> {config.OUTPUT / 'fig_bdc_concentration_lorenz.png'}")

    # ══════════════════════════════════════════════════════════════════
    # LAYER 3: BDC bank-debt vs bank-side L_lines triangulation
    # ══════════════════════════════════════════════════════════════════
    print("\n[6] Layer 3: BDC bank-debt vs bank-side linkage triangulation")
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
    print(f"  Saved -> {config.OUTPUT / 'fig_bdc_vs_bank_linkage.png'}")

    if len(combined) > 5:
        corr_triangulate = combined[["sum_bank_debt", "L_lines"]].corr().iloc[0, 1]
        print(f"  Correlation, LEVELS (BDC aggregate bank_debt, bank-side mean L_lines): {corr_triangulate:.4f}")
        log.append(f"\nTriangulation correlation, LEVELS (BDC bank_debt, bank-side L_lines): {corr_triangulate:.4f}")

        # Levels of both series likely trend upward over the sample from general
        # private-credit-sector growth -- a raw level correlation between two
        # trending series is a classic spurious-correlation risk. Check the
        # quarter-on-quarter CHANGE in each series instead, which nets out any
        # shared trend and asks the sharper question: do they move together
        # quarter to quarter, not just grow together over 15 years?
        combined_sorted = combined.sort_values("quarter_end").copy()
        combined_sorted["d_bank_debt"] = combined_sorted["sum_bank_debt"].diff()
        combined_sorted["d_L_lines"] = combined_sorted["L_lines"].diff()
        diffed = combined_sorted.dropna(subset=["d_bank_debt", "d_L_lines"])
        if len(diffed) > 5:
            corr_diff = diffed[["d_bank_debt", "d_L_lines"]].corr().iloc[0, 1]
            print(f"  Correlation, QoQ CHANGES (detrended, avoids spurious-correlation risk): {corr_diff:.4f}")
            log.append(f"Triangulation correlation, QoQ CHANGES (detrended): {corr_diff:.4f}")
            log.append("  (Levels correlation can be spurious if both series simply trend upward")
            log.append("  together over the sample; the QoQ-change correlation is the more")
            log.append("  defensible number to cite, since it nets out any shared trend.)")

    # ── Save log ──────────────────────────────────────────────────────────
    (config.OUTPUT / "bdc_supplement_log.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\n  Log -> {config.OUTPUT / 'bdc_supplement_log.txt'}")
    print("\n  Done.")


if __name__ == "__main__":
    main()