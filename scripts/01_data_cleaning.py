"""
01_data_cleaning.py

Loads the two WRDS Call Report pulls, dedupes, merges, and applies the sample-definition filters. Produces the clean bank-quarter panel that
02_linkage_index.py builds L_it on top of. No ratio construction here, just a reliably filtered sample.

INPUT   data/raw/rcfd1.csv   (rcfd3820, rcfd1764, rcfd1763, rcfd3433, rcfd3819, rcfdj454, pv-series)
        data/raw/rcfd2.csv   (rcfdj457/458/459, rcfd2170, rcfd3210, rcfd3818, rcfd3817)
        data/raw/rconj454.csv
OUTPUT  data/clean/panel_raw.parquet
        outputs/cleaning_log.txt

STEPS
    1. Load both files, drop exact-duplicate rows independently (WRDS export artifact, ~2% of rows in each file)
    2. Merge on (rssd9001, rssd9999); merge in rconj454
    3. Trim to analysis window
    4. Keep FFIEC 031 filers only
    5. Minimum asset size filter ($300m)
    6. Drop missing total assets
    7. Flag and drop merger quarters (>50% one-quarter asset jump)
    8. Zero-fill Schedule RC-L items
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
config = import_module("00_config")


def _log(log, label, df):
    log.append(
        f"{label:<60} "
        f"rows={len(df):>8,}  "
        f"banks={df['rssd9001'].nunique():>5,}"
    )


def _load_and_dedupe(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    n_before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df, n_before - len(df)


def main():
    log = [
        "DATA CLEANING AUDIT LOG",
        "=" * 65,
        f"{'Step':<60} {'rows':>8}  {'banks':>5}",
        "-" * 65,
    ]

    df1, dropped1 = _load_and_dedupe(config.F_RCFD1)
    df2, dropped2 = _load_and_dedupe(config.F_RCFD2)
    log.append(f"  rcfd1.csv: dropped {dropped1:,} exact-duplicate rows (WRDS export artifact)")
    log.append(f"  rcfd2.csv: dropped {dropped2:,} exact-duplicate rows (WRDS export artifact)")

    for df in (df1, df2):
        df["rssd9999"] = pd.to_datetime(df["rssd9999"], dayfirst=True, errors="coerce")
        df["rssd9999"] = df["rssd9999"].dt.to_period("Q").dt.to_timestamp("Q")

    # Keys duplicated with non-identical values (amended/restated filings) are dropped from both files #
    conflict_keys_1 = set(
        zip(df1.loc[df1.duplicated(subset=config.ID_COLS, keep=False), "rssd9001"],
            df1.loc[df1.duplicated(subset=config.ID_COLS, keep=False), "rssd9999"])
    )
    conflict_keys_2 = set(
        zip(df2.loc[df2.duplicated(subset=config.ID_COLS, keep=False), "rssd9001"],
            df2.loc[df2.duplicated(subset=config.ID_COLS, keep=False), "rssd9999"])
    )
    conflict_keys = conflict_keys_1 | conflict_keys_2
    log.append(f"  Dropped {len(conflict_keys)} (bank,quarter) keys with conflicting "
               f"(non-identical) duplicate rows in either file -- likely amended/"
               f"restated filings; no submission-date field available to pick the "
               f"correct version.")

    key_series_1 = pd.Series(list(zip(df1["rssd9001"], df1["rssd9999"])), index=df1.index)
    key_series_2 = pd.Series(list(zip(df2["rssd9001"], df2["rssd9999"])), index=df2.index)
    df1 = df1[~key_series_1.isin(conflict_keys)].copy()
    df2 = df2[~key_series_2.isin(conflict_keys)].copy()

    k1 = set(zip(df1["rssd9001"], df1["rssd9999"]))
    k2 = set(zip(df2["rssd9001"], df2["rssd9999"]))
    assert k1 == k2, (
        f"Merge keys do not match 1:1 after conflict resolution -- "
        f"{len(k1 - k2)} only in rcfd1, {len(k2 - k1)} only in rcfd2."
    )

    # df2 duplicates rssd9017 / rssdfininstfilingtype from df1 #
    df2_data_cols = [c for c in df2.columns if c not in ["rssd9017", "rssdfininstfilingtype"]]
    df = df1.merge(df2[df2_data_cols], on=config.ID_COLS, how="inner")
    _log(log, "2. Merged rcfd1 + rcfd2 (inner join on bank,quarter)", df)

    # Domestic-office loans to NDFI - resolves the RCFD1520 gap #
    rj454 = pd.read_csv(config.F_RCONJ454, low_memory=False)
    rj454.columns = rj454.columns.str.lower().str.strip()
    rj454 = rj454.drop_duplicates(keep="first")
    rj454["rssd9999"] = pd.to_datetime(rj454["rssd9999"], dayfirst=True, errors="coerce")
    rj454["rssd9999"] = rj454["rssd9999"].dt.to_period("Q").dt.to_timestamp("Q")

    dup_mask = rj454.duplicated(subset=config.ID_COLS, keep=False)
    n_conflict_keys = rj454.loc[dup_mask, config.ID_COLS].drop_duplicates().shape[0]
    if n_conflict_keys > 0:
        log.append(f"rconj454: {n_conflict_keys} conflicting keys set to NaN")
        rj454 = rj454[~dup_mask].copy()
    rj454 = rj454[config.ID_COLS + ["rconj454"]]

    n_before_merge = len(df)
    df = df.merge(rj454, on=config.ID_COLS, how="left")
    assert len(df) == n_before_merge, "rconj454 merge changed row count -- check for duplicates"
    n_matched = df["rconj454"].notna().sum()
    log.append(f"rconj454 merged: {n_matched:,} of {len(df):,} rows matched")

    df = df[
        (df["rssd9999"] >= config.START_DATE) &
        (df["rssd9999"] <= config.END_DATE)
    ].copy()
    _log(log, f"3. Trim to analysis window ({config.START_DATE} to {config.END_DATE})", df)

    log.append("")
    log.append("Filing type distribution (pre-filter):")
    for ft, cnt in df["rssdfininstfilingtype"].value_counts(dropna=False).items():
        log.append(f"  Type {ft:>3}: {cnt:>10,} rows")

    df = df[df["rssdfininstfilingtype"] == config.FILING_TYPE].copy()
    _log(log, "5. Keep FFIEC 031 filers only", df)

    df = df[df["rcfd2170"] >= config.MIN_ASSETS_USD].copy()
    _log(log, f"6. Assets >= $300m ({config.MIN_ASSETS_USD:,}k)", df)

    before = len(df)
    df = df[df["rcfd2170"].notna()].copy()
    _log(log, f"7. Drop missing rcfd2170 (removed {before - len(df):,})", df)

    df = df.sort_values(["rssd9001", "rssd9999"]).reset_index(drop=True)
    df["assets_lag"] = df.groupby("rssd9001")["rcfd2170"].shift(1)
    df["asset_growth"] = (df["rcfd2170"] - df["assets_lag"]) / df["assets_lag"]
    df["merger_flag"] = (df["asset_growth"] > config.MERGER_JUMP_THRESHOLD) & df["assets_lag"].notna()
    n_flagged = df["merger_flag"].sum()
    log.append(f"  Merger quarters flagged and dropped: {n_flagged:,}")
    df = df[~df["merger_flag"]].copy()
    _log(log, "8. Drop merger quarters", df)

    # Confirmed against the final panel, (which gives different, misleading counts) #
    n_2005q2 = (df["rssd9999"] == pd.Timestamp("2005-06-30")).sum()
    n_2005q3 = (df["rssd9999"] == pd.Timestamp("2005-09-30")).sum()
    n_2005q4 = (df["rssd9999"] == pd.Timestamp("2005-12-31")).sum()
    log.append(f"  NOTE: 2005Q3 shows {n_2005q3} banks vs {n_2005q2}/{n_2005q4} in "
               f"surrounding quarters. Q3 2005 was the first quarter Call Reports were "
               f"processed through the newly-implemented CDR electronic filing system, "
               f"and Hurricane Katrina struck during the same reporting quarter -- the "
               f"agencies formally acknowledged resulting filing delays (FDIC/OCC/Fed "
               f"joint filing FIL-94-2005, Sept 21 2005). Left as a natural panel gap, "
               f"not patched.")

    for col in config.RC_L_ZERO_FILL_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            if n > 0:
                log.append(f"  {col}: {n:,} NaN -> 0 (RC-L zero-fill)")

    # rcfd3818 and the NDFI-breakdown items (rcfdj454, pv-series) #

    pv_cols = [c for c in df.columns if c.startswith("rcfdpv")]
    keep = (
        ["rssd9001", "rssd9999", "rssd9017", "rssdfininstfilingtype"]
        + ["rcfd3819", "rcfd3433", "rcfd3820", "rcfd1763", "rcfd1764"]
        + ["rcfdj454", "rconj454"]
        + pv_cols
        + ["rcfdj457", "rcfdj458", "rcfdj459", "rcfd2170", "rcfd3210", "rcfd3818", "rcfd3817"]
        + ["merger_flag", "asset_growth"]
    )
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values(["rssd9001", "rssd9999"]).reset_index(drop=True)
    _log(log, "10. Final clean panel", df)

    df.to_parquet(config.F_PANEL_RAW, index=False)

    log += ["", "-" * 65, "FINAL VARIABLE COVERAGE:", ""]
    for col in ["rcfd3819", "rcfd3433", "rcfdj458", "rconj454", "rcfd2170", "rcfd3210",
                "rcfd1763", "rcfd1764", "rcfdj454"]:
        if col in df.columns:
            nn = df[col].notna().sum()
            pct = nn / len(df) * 100
            log.append(f"  {col:<20} {pct:>5.1f}%  ({nn:,} / {len(df):,})")

    (config.OUTPUT / "cleaning_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()