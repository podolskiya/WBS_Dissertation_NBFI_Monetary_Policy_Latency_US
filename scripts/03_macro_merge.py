"""
03_macro_merge.py
------------------
Loads the manually-downloaded FRED CSVs and the Jarocinski-Karadi
monetary policy shock series, aggregates both to quarterly frequency,
and merges into the clean panel.

FILES
    NFCI.csv             https://fred.stlouisfed.org/series/NFCI
    NFCICREDIT.csv       https://fred.stlouisfed.org/series/NFCICREDIT
    DGS10.csv            https://fred.stlouisfed.org/series/DGS10
    FEDFUNDS.csv         https://fred.stlouisfed.org/series/FEDFUNDS
    shocks_fed_jk_m.csv  https://marekjarocinski.github.io/jkshocks/jkshocks.html

INPUT   data/clean/panel_clean.parquet
        data/raw/NFCI.csv, NFCICREDIT.csv, DGS10.csv, FEDFUNDS.csv
        data/raw/shocks_fed_jk_m.csv

OUTPUT  data/clean/panel_macro.parquet / .csv
        data/clean/macro_quarterly.csv  (for inspection)
        outputs/macro_merge_log.txt

VARIABLES CONSTRUCTED
    nfci, nfci_credit    NFCI level and credit sub-index
    dgs10, fedfunds      10yr Treasury yield, Fed funds rate (pct)
    term_spread          dgs10 - fedfunds
    nfci_change          quarter-on-quarter change in NFCI
    mp_shock, cbi_shock  quarterly-summed Jarocinski-Karadi shocks
                         (mp = monetary policy, cbi = central bank
                         information -- confound/decomposition)
    mp_shock_median, cbi_shock_median   median-rotation robustness variant

FREQUENCY MISMATCH (*1)
"""
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
config = import_module("00_config")

warnings.filterwarnings("ignore", category=FutureWarning)


def _load_fred_csv(path: Path, value_col: str) -> pd.Series:
    if not path.exists():
        print(f"  MISSING: {path}")
        return pd.Series(dtype=float, name=value_col)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["date", value_col]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[value_col]
    df = pd.to_numeric(df.replace(".", np.nan), errors="coerce")
    print(f"  Loaded {path.name:<16} {df.notna().sum():>5} obs  "
          f"{df.first_valid_index().date()} to {df.last_valid_index().date()}")
    return df


def _load_shocks(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  MISSING: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["rssd9999"] = _to_quarter_end(pd.DatetimeIndex(df["date"]))
    q = (df.groupby("rssd9999")
           .agg(mp_shock=("MP_pm", "sum"),
                cbi_shock=("CBI_pm", "sum"),
                mp_shock_median=("MP_median", "sum"),
                cbi_shock_median=("CBI_median", "sum"))
           .reset_index())
    print(f"  Loaded {path.name:<20} {len(df)} monthly obs -> {len(q)} quarters "
          f"({q['rssd9999'].min().date()} to {q['rssd9999'].max().date()})")
    return q


def _to_quarter_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_period("Q").to_timestamp("Q")


def main():
    print("\n" + "=" * 65)
    print("03_macro_merge.py  --  loading and merging macro variables")
    print("=" * 65)

    log = ["MACRO MERGE AUDIT LOG", "=" * 65, ""]

    # [1] Load FRED CSVs
    print("\n[1] Loading FRED CSV files from data/raw/")
    series_files = {
        config.F_NFCI:       "nfci",
        config.F_NFCICREDIT: "nfci_credit",
        config.F_DGS10:      "dgs10",
        config.F_FEDFUNDS:   "fedfunds",
    }
    frames, missing = {}, []
    for path, col in series_files.items():
        s = _load_fred_csv(path, col)
        if s.empty:
            missing.append(path.name)
        else:
            frames[col] = s
    if missing:
        print(f"\n  WARNING: {len(missing)} file(s) missing: {missing}")
        log.append(f"WARNING: missing files: {missing}")
    if not frames:
        print("\n  ERROR: no macro files found. Aborting.")
        return

    # [2] Combine into wide frame
    print("\n[2] Combining into wide frame")
    macro_daily = pd.DataFrame(frames).sort_index()
    print(f"  Combined shape: {macro_daily.shape}")
    log.append(f"Combined raw frame shape: {macro_daily.shape}")

    # [3] Aggregate to quarter-end (mean within quarter)
    print("\n[3] Aggregating to quarterly (mean within quarter)")
    macro_q = macro_daily.copy()
    macro_q.index = _to_quarter_end(macro_q.index)
    macro_q = macro_q.groupby(macro_q.index).mean()
    macro_q.index.name = "rssd9999"
    macro_q = macro_q.reset_index()
    log.append("")
    log.append("NOTE ON FREQUENCY MISMATCH: (*1)")

    # [4] Derived variables
    print("\n[4] Constructing derived macro variables")
    if "dgs10" in macro_q.columns and "fedfunds" in macro_q.columns:
        macro_q["term_spread"] = macro_q["dgs10"] - macro_q["fedfunds"]
        print("  term_spread = dgs10 - fedfunds")
    if "nfci" in macro_q.columns:
        macro_q = macro_q.sort_values("rssd9999")
        macro_q["nfci_change"] = macro_q["nfci"].diff()
        print("  nfci_change = quarter-on-quarter change in NFCI")

    macro_q = macro_q[
        (macro_q["rssd9999"] >= config.START_DATE) &
        (macro_q["rssd9999"] <= config.END_DATE)
    ].copy()
    print(f"\n  Quarterly macro series: {len(macro_q)} quarters, "
          f"{macro_q['rssd9999'].min().date()} to {macro_q['rssd9999'].max().date()}")

    # [4b] Merge monetary policy shock series
    print("\n[4b] Loading Jarocinski-Karadi monetary policy shocks")
    shocks_q = _load_shocks(config.F_SHOCKS_MONTHLY)
    if not shocks_q.empty:
        n_before_shock_merge = len(macro_q)
        macro_q = macro_q.merge(shocks_q, on="rssd9999", how="left")
        assert len(macro_q) == n_before_shock_merge, "Shock merge changed row count -- investigate"
        n_missing_shock = macro_q["mp_shock"].isna().sum()
        print(f"  Merged: {len(macro_q)} quarters, {n_missing_shock} with no shock data")
        log.append(f"\nShock series merged: {n_missing_shock} quarters with no shock "
                   f"data (outside JK sample window)")
    else:
        log.append("\nWARNING: shock series file not found, mp_shock/cbi_shock will be all-NaN")

    macro_q.to_csv(config.F_MACRO_QUARTERLY, index=False)
    print(f"  Saved -> {config.F_MACRO_QUARTERLY}")

    # [5] Merge with clean panel
    print("\n[5] Merging with clean panel")
    df = pd.read_parquet(config.F_PANEL_CLEAN)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])
    n_before = len(df)
    df = df.merge(macro_q, on="rssd9999", how="left")
    assert len(df) == n_before, (
        f"Row count changed after merge ({n_before} -> {len(df)}) -- "
        f"macro_q must have duplicate quarter-end dates."
    )
    print(f"  Panel rows: {n_before:,} -> {len(df):,} (unchanged, as expected)")
    log.append(f"\nPanel merge: {n_before:,} rows in, {len(df):,} rows out (unchanged)")

    # [6] Coverage check
    print("\n[6] Macro variable coverage in merged panel")
    check_cols = ["nfci", "nfci_credit", "dgs10", "fedfunds", "term_spread", "nfci_change",
                  "mp_shock", "cbi_shock", "mp_shock_median", "cbi_shock_median"]
    log.append("\nMacro variable coverage in merged panel:")
    for col in check_cols:
        if col in df.columns:
            nn, pct = df[col].notna().sum(), df[col].notna().sum() / len(df) * 100
            mn, mx, med = df[col].min(), df[col].max(), df[col].median()
            line = f"  {col:<12} {pct:>5.1f}%  median={med:>7.4f}  [{mn:.4f}, {mx:.4f}]"
            print(line)
            log.append(line)

    # [7] Correlations with L_it
    print("\n[7] Correlations with L_it")
    corr_cols = ["L_it"] + [c for c in check_cols if c in df.columns]
    corr = df[corr_cols].corr()["L_it"].drop("L_it")
    log.append("\nCorrelations with L_it:")
    for col, val in corr.items():
        line = f"  corr(L_it, {col:<12}) = {val:>7.4f}"
        print(f"  {line}")
        log.append(line)

    # [8] Save
    print("\n[8] Saving")
    df.to_parquet(config.F_PANEL_MACRO, index=False)
    df.to_csv(config.F_PANEL_MACRO_CSV, index=False)
    print(f"  Parquet -> {config.F_PANEL_MACRO}")
    print(f"  CSV     -> {config.F_PANEL_MACRO_CSV}")
    print(f"  Final panel shape: {df.shape}")

    log_text = "\n".join(log)
    (config.OUTPUT / "macro_merge_log.txt").write_text(log_text, encoding="utf-8")
    print(f"\n  Log -> {config.OUTPUT / 'macro_merge_log.txt'}")


if __name__ == "__main__":
    main()

# ============================================================
# APPENDIX
# ============================================================
# *1  NFCI/NFCICREDIT (weekly), DGS10 (daily), FEDFUNDS (monthly) all
#     aggregated to quarterly means to match the bank-quarter panel.
#     Discards within-quarter timing -- standard in the bank-lending
#     literature, but the same limitation applies to the shock series,
#     which is the actual regressor of interest in Proposition 1. The
#     local projections in 04_regression_p1.py identify the effect of
#     the quarterly-cumulated shock, not a single FOMC surprise. State
#     this explicitly in the methodology write-up.