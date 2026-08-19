"""
02_linkage_index.py
--------------------
Constructs the bank-NBFI linkage index (L_it) and variants from the
cleaned panel, runs validation checks, and builds the concentration
diagnostic (requested by supervisor -- is L_it dominated by a handful
of very large banks?).

INPUT   data/clean/panel_raw.parquet

OUTPUT  data/clean/panel_clean.parquet / .csv
        outputs/linkage_index_log.txt
        outputs/table_lit_distribution.txt
        outputs/table_concentration.txt
        outputs/fig_lit_timeseries.png
        outputs/fig_lit_components.png
        outputs/fig_lit_by_size_decile.png
        outputs/fig_concentration_lorenz.png

L_it = (rcfd3819 + rcfd3433 + rcfdj458 + rconj454) / rcfd2170
    rcfd3819  Financial standby letters of credit
    rcfd3433  Securities lent with cash collateral
    rcfdj458  Unused commitments to FIs (zero pre-2010)
    rconj454  Loans to nondepository financial institutions
    rcfd2170  Total assets (denominator)
"""
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def _winsorise(series, lower=config.WINSOR_LOWER, upper=config.WINSOR_HIGH):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def _log(log, label, df):
    entry = f"{label:<55} rows={len(df):>8,}  banks={df['rssd9001'].nunique():>5,}"
    log.append(entry)
    print(f"  {entry}")


def main():
    print("\n" + "=" * 65)
    print("02_linkage_index.py  --  constructing L_it and validating")
    print("=" * 65)

    log = ["LINKAGE INDEX AUDIT LOG", "=" * 65, ""]

    # [0] Load
    print("\n[0] Loading panel_raw.parquet")
    df = pd.read_parquet(config.F_PANEL_RAW)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])
    _log(log, "0. Raw clean panel loaded", df)

    # [1] Winsorise components
    print(f"\n[1] Winsorising components at [{config.WINSOR_LOWER:.0%}, {config.WINSOR_HIGH:.0%}]")
    for col in ["rcfdj458", "rcfd3819", "rcfd3433", "rconj454", "rcfd2170",
                "rcfd3210", "rcfd1763", "rcfd1764"]:
        if col in df.columns and df[col].notna().sum() > 10:
            lo = df[col].quantile(config.WINSOR_LOWER)
            hi = df[col].quantile(config.WINSOR_HIGH)
            df[col] = df[col].clip(lower=lo, upper=hi)
            print(f"  {col:<12} [{lo:>14,.0f}, {hi:>14,.0f}]")

    # [2] Build L_it and variants
    print("\n[2] Building L_it and variants")
    denom = df["rcfd2170"]
    df["L_it"] = (df["rcfd3819"].fillna(0) + df["rcfd3433"].fillna(0)
                  + df["rcfdj458"].fillna(0) + df["rconj454"].fillna(0)) / denom
    df["L_middle"]     = (df["rcfd3819"].fillna(0) + df["rcfd3433"].fillna(0)) / denom
    df["L_guarantees"] = df["rcfd3819"].fillna(0) / denom
    df["L_securities"] = df["rcfd3433"].fillna(0) / denom
    df["L_lines"]      = df["rcfdj458"].fillna(0) / denom
    df["L_ndfi_loans"] = df["rconj454"].fillna(0) / denom
    for col in ["L_it", "L_middle", "L_guarantees", "L_securities", "L_lines", "L_ndfi_loans"]:
        df[col] = df[col].clip(lower=0)

    print("[3] Winsorising L_it and variants at 99th percentile")
    for col in ["L_it", "L_middle", "L_guarantees", "L_securities", "L_lines", "L_ndfi_loans"]:
        hi = df[col].quantile(config.WINSOR_HIGH)
        n_clip = (df[col] > hi).sum()
        df[col] = df[col].clip(upper=hi)
        print(f"  {col:<15} 99th={hi:.6f}  clipped {n_clip:,} obs")

    log.append("L_it = (rcfd3819 + rcfd3433 + rcfdj458 + rconj454) / rcfd2170")
    log.append("rcfdj458, rconj454 = 0 for 2005-2009 (item didn't exist pre-2010)")

    # [4] Controls and outcome variables
    print("\n[4] Constructing controls and outcome variables")
    df["capital_ratio"] = _winsorise(df["rcfd3210"] / df["rcfd2170"])
    df["log_assets"]    = np.log(df["rcfd2170"])
    df["year"]          = df["rssd9999"].dt.year
    df["quarter"]       = df["rssd9999"].dt.quarter
    df["year_quarter"]  = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
    df["post_2010"]     = (df["year"] >= 2010).astype(int)
    df = df.sort_values(["rssd9001", "rssd9999"])
    ci = df["rcfd1763"].fillna(0) + df["rcfd1764"].fillna(0)
    ci_lag = ci.groupby(df["rssd9001"]).shift(1)
    df["ci_loan_growth"] = ((ci - ci_lag) / ci_lag).clip(lower=-1, upper=3)
    df["has_sec_lending"] = (df["rcfd3433"] > 0).astype(int)

    # [5] credit_growth_4q -- calendar-based self-merge (robust to panel gaps)
    print("[5] Constructing credit_growth_4q (calendar-merge, h=4)")
    df["ci_level"] = ci
    df["rssd9999_fwd"] = df["rssd9999"] + pd.DateOffset(months=9)
    df["rssd9999_fwd"] = df["rssd9999_fwd"] + pd.offsets.QuarterEnd(0)
    df_fwd = df[["rssd9001", "rssd9999", "ci_level"]].rename(
        columns={"rssd9999": "rssd9999_fwd", "ci_level": "ci_level_fwd4"}
    )
    df = df.merge(df_fwd, on=["rssd9001", "rssd9999_fwd"], how="left")
    df = df.drop(columns=["rssd9999_fwd"])
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.log(df["ci_level_fwd4"] / df["ci_level"])
    df["credit_growth_4q"] = raw.replace([np.inf, -np.inf], np.nan)
    lo4, hi4 = df["credit_growth_4q"].quantile([0.01, 0.99])
    df["credit_growth_4q"] = df["credit_growth_4q"].clip(lower=lo4, upper=hi4)
    n_valid = df["credit_growth_4q"].notna().sum()
    print(f"  Valid obs: {n_valid:,} of {len(df):,}")
    log.append(f"credit_growth_4q: {n_valid:,} valid obs of {len(df):,}, "
               f"winsorised [{lo4:.4f}, {hi4:.4f}]")

    # [6] Save clean panel
    print("\n[6] Saving clean panel")
    keep = [
        "rssd9001", "rssd9999", "rssd9017", "year", "quarter", "year_quarter",
        "post_2010", "L_it", "L_middle", "L_guarantees", "L_securities", "L_lines", "L_ndfi_loans",
        "rcfd3819", "rcfd3433", "rcfdj458", "rcfdj457", "rcfdj459", "rcfd3820", "rconj454",
        "rcfd2170", "rcfd3210", "capital_ratio", "log_assets", "ci_loan_growth",
        "rcfd1763", "rcfd1764", "merger_flag", "asset_growth", "has_sec_lending",
        "rcfd3817", "credit_growth_4q", "ci_level", "rcfdj454",
    ] + [c for c in df.columns if c.startswith("rcfdpv")]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values(["rssd9001", "rssd9999"]).reset_index(drop=True)
    _log(log, "Final clean panel with L_it", df)
    df.to_parquet(config.F_PANEL_CLEAN, index=False)
    df.to_csv(config.F_PANEL_CSV, index=False)
    print(f"  Saved -> {config.F_PANEL_CLEAN}")
    print(f"  Saved -> {config.F_PANEL_CSV}")

    # [7] Validation exhibits
    print("\n[7] Validation exhibits")
    val_lines = ["L_it VALIDATION SUMMARY", "=" * 60]

    ts = (df.groupby("rssd9999")
            .agg(mean_Lit=("L_it", "mean"), median_Lit=("L_it", "median"),
                 p25_Lit=("L_it", lambda x: x.quantile(0.25)),
                 p75_Lit=("L_it", lambda x: x.quantile(0.75)),
                 n_banks=("rssd9001", "nunique"))
            .reset_index())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ts["rssd9999"], ts["mean_Lit"], color=NAVY, lw=2, label="Cross-sectional mean")
    ax.plot(ts["rssd9999"], ts["median_Lit"], color=BLUE, lw=1.5, ls="--", label="Median")
    ax.fill_between(ts["rssd9999"], ts["p25_Lit"], ts["p75_Lit"], color=BLUE, alpha=0.12, label="IQR")
    ax.set_xlabel("Quarter"); ax.set_ylabel("L_it (ratio to total assets)")
    ax.set_title(f"Cross-sectional mean of L_it -- FFIEC 031 banks, {df['year'].min()}-{df['year'].max()}")
    ax.legend(frameon=False, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_lit_timeseries.png", bbox_inches="tight")
    plt.close()
    val_lines += [
        "", "CHECK 1: Time-series (cross-sectional mean L_it)",
        f"  Min quarterly mean: {ts['mean_Lit'].min():.6f} ({ts.loc[ts['mean_Lit'].idxmin(),'rssd9999'].date()})",
        f"  Max quarterly mean: {ts['mean_Lit'].max():.6f} ({ts.loc[ts['mean_Lit'].idxmax(),'rssd9999'].date()})",
        f"  Full-period mean:   {ts['mean_Lit'].mean():.6f}",
    ]

    for col in ["rcfd3819", "rcfd3433", "rcfdj458", "rconj454"]:
        df[f"{col}_ratio"] = df[col] / df["rcfd2170"]
    ts2 = (df.groupby("rssd9999")
             .agg(sblc=("rcfd3819_ratio", "mean"), seclnd=("rcfd3433_ratio", "mean"),
                  lines=("rcfdj458_ratio", "mean"), ndfi=("rconj454_ratio", "mean")).reset_index())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(ts2["rssd9999"], ts2["sblc"], ts2["seclnd"], ts2["lines"], ts2["ndfi"],
                 labels=["Financial standby LCs", "Securities lent", "Credit lines to FIs (2010+)",
                        "Loans to NDFI (2010+)"],
                 colors=[NAVY, BLUE, GREEN, RED], alpha=0.75)
    ax.axvline(pd.Timestamp("2010-01-01"), color="gray", lw=1, ls="--")
    ax.set_xlabel("Quarter"); ax.set_ylabel("Mean ratio to total assets")
    ax.set_title("L_it components over time")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_lit_components.png", bbox_inches="tight")
    plt.close()

    bank_avg = df.groupby("rssd9001")["rcfd2170"].mean().reset_index()
    bank_avg["size_decile"] = pd.qcut(bank_avg["rcfd2170"], q=10, labels=range(1, 11))
    df = df.merge(bank_avg[["rssd9001", "size_decile"]], on="rssd9001", how="left")
    decile_stats = (df.groupby("size_decile", observed=True)["L_it"]
                      .agg(["mean", "median", "std", "count"]).reset_index())
    decile_stats.columns = ["decile", "mean", "median", "std", "n_obs"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(decile_stats["decile"].astype(int), decile_stats["mean"], color=NAVY, alpha=0.85)
    ax.set_xlabel("Asset size decile (1=smallest, 10=largest)"); ax.set_ylabel("Mean L_it")
    ax.set_title("Mean L_it by bank-size decile")
    ax.set_xticks(range(1, 11))
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_lit_by_size_decile.png", bbox_inches="tight")
    plt.close()
    val_lines += ["", "CHECK 2: L_it by asset size decile",
                  f"{'Decile':<8}{'Mean':>12}{'Median':>12}{'N obs':>10}", "-" * 42]
    for _, row in decile_stats.iterrows():
        val_lines.append(f"  {int(row['decile']):<6}{row['mean']:>12.6f}{row['median']:>12.6f}{int(row['n_obs']):>10,}")

    desc = df["L_it"].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99])
    val_lines += ["", "CHECK 3: L_it distribution (full sample)", desc.to_string()]
    val_text = "\n".join(val_lines)
    (config.OUTPUT / "table_lit_distribution.txt").write_text(val_text, encoding="utf-8")
    print(f"  Saved -> {config.OUTPUT / 'table_lit_distribution.txt'}")

    # [8] Concentration diagnostic
    print("\n[8] Concentration diagnostic")
    conc_lines = [
        "CONCENTRATION DIAGNOSTIC",
        "Checks whether NBFI-linkage exposure is dominated by a small",
        "number of very large banks.",
        "=" * 60, "",
    ]
    df["numerator"] = (df["rcfd3819"].fillna(0) + df["rcfd3433"].fillna(0)
                       + df["rcfdj458"].fillna(0) + df["rconj454"].fillna(0))
    shares, hhis = [], []
    for q, g in df.groupby("rssd9999"):
        g = g.sort_values("numerator", ascending=False)
        total = g["numerator"].sum()
        if total <= 0:
            continue
        shares.append({
            "q": q,
            "top5":  g["numerator"].head(5).sum() / total,
            "top10": g["numerator"].head(10).sum() / total,
            "top20": g["numerator"].head(20).sum() / total,
            "n_banks": g["rssd9001"].nunique(),
        })
        hhis.append(((g["numerator"] / total) ** 2).sum())
    sh = pd.DataFrame(shares)
    conc_lines += [
        "Average share of aggregate NBFI-linkage dollar exposure held by:",
        f"  Top 5 banks (of ~{sh.n_banks.mean():.0f}):  {sh.top5.mean()*100:.1f}%  "
        f"(range {sh.top5.min()*100:.1f}-{sh.top5.max()*100:.1f}%)",
        f"  Top 10 banks:                {sh.top10.mean()*100:.1f}%  "
        f"(range {sh.top10.min()*100:.1f}-{sh.top10.max()*100:.1f}%)",
        f"  Top 20 banks:                {sh.top20.mean()*100:.1f}%  "
        f"(range {sh.top20.min()*100:.1f}-{sh.top20.max()*100:.1f}%)",
        f"  Mean HHI (dollar exposure):  {np.mean(hhis):.4f}  "
        f"(uniform benchmark = {1/sh.n_banks.mean():.4f})",
        "",
    ]
    bank_avg2 = df.groupby("rssd9001").agg(mean_Lit=("L_it", "mean"),
                                            mean_assets=("rcfd2170", "mean")).reset_index()
    corr = bank_avg2[["mean_Lit", "mean_assets"]].corr().iloc[0, 1]
    n = len(bank_avg2)
    top10pct = max(1, int(n * 0.10))
    bank_avg2_sorted = bank_avg2.sort_values("mean_assets", ascending=False)
    top_decile_Lit = bank_avg2_sorted.head(top10pct)["mean_Lit"].mean()
    rest_Lit = bank_avg2_sorted.tail(n - top10pct)["mean_Lit"].mean()
    conc_lines += [
        f"Correlation (bank-level mean L_it, bank-level mean assets): {corr:.3f}",
        f"  -> {'ratio itself concentrated in largest banks' if abs(corr) > 0.3 else 'ratio NOT strongly concentrated in largest banks despite dollar concentration above'} (*1)",
        "",
        f"N banks: {n}. Top 10% by assets ({top10pct} banks): mean L_it = {top_decile_Lit:.4f}",
        f"Bottom 90% by assets ({n - top10pct} banks): mean L_it = {rest_Lit:.4f}",
    ]

    bank_total = df.groupby("rssd9001")["numerator"].sum().sort_values()
    cum_banks = np.arange(1, len(bank_total) + 1) / len(bank_total)
    cum_exposure = bank_total.cumsum() / bank_total.sum()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="Perfect equality")
    ax.plot(cum_banks, cum_exposure, color=RED, lw=2, label="Lorenz curve (L_it numerator)")
    ax.fill_between(cum_banks, cum_exposure, cum_banks, color=RED, alpha=0.1)
    ax.set_xlabel("Cumulative share of banks (smallest to largest)")
    ax.set_ylabel("Cumulative share of aggregate NBFI-linkage exposure")
    ax.set_title("Concentration of bank-NBFI linkage exposure")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_concentration_lorenz.png", bbox_inches="tight")
    plt.close()
    conc_text = "\n".join(conc_lines)
    (config.OUTPUT / "table_concentration.txt").write_text(conc_text, encoding="utf-8")
    print(f"  Saved -> {config.OUTPUT / 'table_concentration.txt'}")
    print("\n" + conc_text)

    log_text = "\n".join(log)
    (config.OUTPUT / "linkage_index_log.txt").write_text(log_text, encoding="utf-8")
    print(f"\n  Log -> {config.OUTPUT / 'linkage_index_log.txt'}")


if __name__ == "__main__":
    main()

# ============================================================
# APPENDIX
# ============================================================
# *1  0.3 is an arbitrary but standard rule-of-thumb cutoff for "not a
#     strong correlation" in this kind of diagnostic -- not derived from
#     anything, just a reasonable line to draw for the write-up.