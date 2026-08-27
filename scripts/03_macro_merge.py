"""
03_macro_merge.py

Loads the manually-downloaded FRED CSVs and the Jarocinski-Karadi monetary policy shock series, aggregates both to quarterly frequency,
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
OUTPUT  data/clean/panel_macro.parquet + .csv
        data/clean/macro_quarterly.csv 
        outputs/macro_merge_log.txt

VARIABLES CONSTRUCTED
    nfci, nfci_credit    NFCI level and credit sub-index
    dgs10, fedfunds      10yr Treasury yield, Fed funds rate (pct)
    term_spread          dgs10 - fedfunds
    nfci_change          quarter-on-quarter change in NFCI
    mp_shock, cbi_shock  quarterly-summed Jarocinski-Karadi shocks
                         Where: (mp = monetary policy, cbi = central bank information -- confound/decomposition)
    mp_shock_median, cbi_shock_median   median-rotation robustness variant
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
        return pd.Series(dtype=float, name=value_col)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["date", value_col]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[value_col]
    df = pd.to_numeric(df.replace(".", np.nan), errors="coerce")
    return df


def _load_shocks(path: Path) -> pd.DataFrame:
    if not path.exists():
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
    return q


def _to_quarter_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_period("Q").to_timestamp("Q")


def main():
    log = ["MACRO MERGE AUDIT LOG", "=" * 65, ""]

    series_files = {
        config.F_NFCI: "nfci",
        config.F_NFCICREDIT: "nfci_credit",
        config.F_DGS10: "dgs10",
        config.F_FEDFUNDS: "fedfunds",
    }
    frames, missing = {}, []
    for path, col in series_files.items():
        s = _load_fred_csv(path, col)
        if s.empty:
            missing.append(path.name)
        else:
            frames[col] = s
    if missing:
        log.append(f"WARNING: missing files: {missing}")
    if not frames:
        print("No macro files found; aborting.")
        return

    macro_daily = pd.DataFrame(frames).sort_index()
    log.append(f"Combined raw frame shape: {macro_daily.shape}")

    macro_q = macro_daily.copy()
    macro_q.index = _to_quarter_end(macro_q.index)
    macro_q = macro_q.groupby(macro_q.index).mean()
    macro_q.index.name = "rssd9999"
    macro_q = macro_q.reset_index()
    log.append("")
    log.append("NOTE ON FREQUENCY MISMATCH: (*1)")

    if "dgs10" in macro_q.columns and "fedfunds" in macro_q.columns:
        macro_q["term_spread"] = macro_q["dgs10"] - macro_q["fedfunds"]
    if "nfci" in macro_q.columns:
        macro_q = macro_q.sort_values("rssd9999")
        macro_q["nfci_change"] = macro_q["nfci"].diff()

    macro_q = macro_q[
        (macro_q["rssd9999"] >= config.START_DATE) &
        (macro_q["rssd9999"] <= config.END_DATE)
    ].copy()

    shocks_q = _load_shocks(config.F_SHOCKS_MONTHLY)
    if not shocks_q.empty:
        n_before_shock_merge = len(macro_q)
        macro_q = macro_q.merge(shocks_q, on="rssd9999", how="left")
        assert len(macro_q) == n_before_shock_merge, "Shock merge changed row count -- investigate"
        n_missing_shock = macro_q["mp_shock"].isna().sum()
        log.append(f"\nShock series merged: {n_missing_shock} quarters with no shock "
                   f"data (outside JK sample window)")
    else:
        log.append("\nWARNING: shock series file not found, mp_shock/cbi_shock will be all-NaN")

    macro_q.to_csv(config.F_MACRO_QUARTERLY, index=False)

    df = pd.read_parquet(config.F_PANEL_CLEAN)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])
    n_before = len(df)
    df = df.merge(macro_q, on="rssd9999", how="left")
    assert len(df) == n_before, (
        f"Row count changed after merge ({n_before} -> {len(df)}) -- "
        f"macro_q must have duplicate quarter-end dates."
    )
    log.append(f"\nPanel merge: {n_before:,} rows in, {len(df):,} rows out (unchanged)")

    check_cols = ["nfci", "nfci_credit", "dgs10", "fedfunds", "term_spread", "nfci_change",
                  "mp_shock", "cbi_shock", "mp_shock_median", "cbi_shock_median"]
    log.append("\nMacro variable coverage in merged panel:")
    for col in check_cols:
        if col in df.columns:
            nn, pct = df[col].notna().sum(), df[col].notna().sum() / len(df) * 100
            mn, mx, med = df[col].min(), df[col].max(), df[col].median()
            log.append(f"  {col:<12} {pct:>5.1f}%  median={med:>7.4f}  [{mn:.4f}, {mx:.4f}]")

    corr_cols = ["L_it"] + [c for c in check_cols if c in df.columns]
    corr = df[corr_cols].corr()["L_it"].drop("L_it")
    log.append("\nCorrelations with L_it:")
    for col, val in corr.items():
        log.append(f"  corr(L_it, {col:<12}) = {val:>7.4f}")

    df.to_parquet(config.F_PANEL_MACRO, index=False)
    df.to_csv(config.F_PANEL_MACRO_CSV, index=False)

    (config.OUTPUT / "macro_merge_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()

# *1  NFCI/NFCICREDIT (weekly), DGS10 (daily), FEDFUNDS (monthly) all aggregated to quarterly means to match the bank-quarter panel.
#     Discards within-quarter timing, which seems to be the standard in the bank-lending literature, but the same limitation applies to the shock series,
#     which is the actual regressor of interest in Proposition 1. The local projections in 04_regression_p1.py identify the effect of
#     the quarterly-cumulated shock, not a single FOMC surprise.