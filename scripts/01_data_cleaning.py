"""
01_data_cleaning.py
--------------------
Loads the two WRDS Call Report pulls, dedupes, merges, and applies the
sample-definition filters. Produces the clean bank-quarter panel that
02_linkage_index.py builds L_it on top of. No ratio construction here --
just getting a reliable, correctly-filtered sample.

INPUT   data/raw/rcfd1.csv   (rcfd3820, rcfd1764, rcfd1763, rcfd3433,
                               rcfd3819, rcfdj454, pv-series)
        data/raw/rcfd2.csv   (rcfdj457/458/459, rcfd2170, rcfd3210,
                               rcfd3818, rcfd3817)
        data/raw/rconj454.csv

OUTPUT  data/clean/panel_raw.parquet
        outputs/cleaning_log.txt

STEPS
    1.  Load both files, drop exact-duplicate rows independently
        (WRDS export artifact, ~2% of rows in each file)
    2.  Merge on (rssd9001, rssd9999); merge in rconj454
    3.  Trim to analysis window
    4.  Keep FFIEC 031 filers only
    5.  Minimum asset size filter ($300m)
    6.  Drop missing total assets
    7.  Flag and drop merger quarters (>50% one-quarter asset jump)
    8.  Zero-fill Schedule RC-L items
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
config = import_module("00_config")


def _log(log, label, df):
    entry = (f"{label:<60} "
             f"rows={len(df):>8,}  "
             f"banks={df['rssd9001'].nunique():>5,}")
    log.append(entry)
    print(f"  {entry}")


def _load_and_dedupe(path, label):
    print(f"\n  Loading {path.name} ...", end=" ")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    n_before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"{n_before:,} rows -> {len(df):,} after dedup ({n_dropped:,} exact duplicates dropped)")
    return df, n_dropped


def main():
    print("\n" + "=" * 65)
    print("01_data_cleaning.py  --  building the clean bank-quarter panel")
    print("=" * 65)

    log = [
        "DATA CLEANING AUDIT LOG",
        "=" * 65,
        f"{'Step':<60} {'rows':>8}  {'banks':>5}",
        "-" * 65,
    ]

    # 1. Load and dedupe both files
    print("\n[1] Loading and deduplicating raw pulls")
    df1, dropped1 = _load_and_dedupe(config.F_RCFD1, "rcfd1")
    df2, dropped2 = _load_and_dedupe(config.F_RCFD2, "rcfd2")
    log.append(f"  rcfd1.csv: dropped {dropped1:,} exact-duplicate rows (WRDS export artifact)")
    log.append(f"  rcfd2.csv: dropped {dropped2:,} exact-duplicate rows (WRDS export artifact)")

    # 2. Merge on (rssd9001, rssd9999)
    print("\n[2] Merging on (rssd9001, rssd9999)")
    for df in (df1, df2):
        df["rssd9999"] = pd.to_datetime(df["rssd9999"], dayfirst=True, errors="coerce")
        df["rssd9999"] = df["rssd9999"].dt.to_period("Q").dt.to_timestamp("Q")

    # Some (bank, quarter) keys are duplicated with DIFFERENT reported
    # values -- most likely original vs. amended filings. WRDS doesn't
    # expose a submission-date field in this pull, so there's no
    # principled way to pick the "correct" version. Dropping these keys
    # from both files keeps the merge a clean 1:1 join, at the cost of
    # ~1% of bank-quarters.
    conflict_keys_1 = set(
        zip(df1.loc[df1.duplicated(subset=config.ID_COLS, keep=False), "rssd9001"],
            df1.loc[df1.duplicated(subset=config.ID_COLS, keep=False), "rssd9999"])
    )
    conflict_keys_2 = set(
        zip(df2.loc[df2.duplicated(subset=config.ID_COLS, keep=False), "rssd9001"],
            df2.loc[df2.duplicated(subset=config.ID_COLS, keep=False), "rssd9999"])
    )
    conflict_keys = conflict_keys_1 | conflict_keys_2
    print(f"  Conflicting (bank,quarter) keys found: {len(conflict_keys_1)} in rcfd1, "
          f"{len(conflict_keys_2)} in rcfd2, {len(conflict_keys)} union -- dropping from both files")
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
    print(f"  Confirmed: {len(k1):,} unique (bank,quarter) keys match 1:1 across both files")

    # df2 carries a second copy of rssd9017 / rssdfininstfilingtype -- drop before merge
    df2_data_cols = [c for c in df2.columns if c not in ["rssd9017", "rssdfininstfilingtype"]]
    df = df1.merge(df2[df2_data_cols], on=config.ID_COLS, how="inner")
    _log(log, "2. Merged rcfd1 + rcfd2 (inner join on bank,quarter)", df)

    # Merge in rconj454 (loans to NDFI, domestic-office basis -- see
    # 00_config.py for why this resolves the RCFD1520 gap)
    print("\n[2b] Merging rconj454")
    rj454 = pd.read_csv(config.F_RCONJ454, low_memory=False)
    rj454.columns = rj454.columns.str.lower().str.strip()
    n_before = len(rj454)
    rj454 = rj454.drop_duplicates(keep="first")
    print(f"  Loaded {n_before:,} rows -> {len(rj454):,} after exact-dup removal")
    rj454["rssd9999"] = pd.to_datetime(rj454["rssd9999"], dayfirst=True, errors="coerce")
    rj454["rssd9999"] = rj454["rssd9999"].dt.to_period("Q").dt.to_timestamp("Q")

    dup_mask = rj454.duplicated(subset=config.ID_COLS, keep=False)
    n_conflict_keys = rj454.loc[dup_mask, config.ID_COLS].drop_duplicates().shape[0]
    if n_conflict_keys > 0:
        print(f"  {n_conflict_keys} conflicting (bank,quarter) keys in rconj454 -- "
              f"set to NaN (same policy as above)")
        log.append(f"rconj454: {n_conflict_keys} conflicting keys set to NaN")
        rj454 = rj454[~dup_mask].copy()
    rj454 = rj454[config.ID_COLS + ["rconj454"]]

    n_before_merge = len(df)
    df = df.merge(rj454, on=config.ID_COLS, how="left")
    assert len(df) == n_before_merge, "rconj454 merge changed row count -- check for duplicates"
    n_matched = df["rconj454"].notna().sum()
    print(f"  Merged: {n_matched:,} of {len(df):,} rows have an rconj454 value "
          f"(rest are pre-2010, where the item doesn't exist)")
    log.append(f"rconj454 merged: {n_matched:,} of {len(df):,} rows matched")

    # 3. Trim to analysis window
    print(f"\n[3] Trimming to {config.START_DATE} - {config.END_DATE}")
    df = df[
        (df["rssd9999"] >= config.START_DATE) &
        (df["rssd9999"] <= config.END_DATE)
    ].copy()
    _log(log, f"3. Trim to analysis window ({config.START_DATE} to {config.END_DATE})", df)

    # 4. Filing type distribution
    print("\n[4] Filing type distribution (all filers, pre-filter)")
    print(df["rssdfininstfilingtype"].value_counts(dropna=False).to_string())
    log.append("")
    log.append("Filing type distribution (pre-filter):")
    for ft, cnt in df["rssdfininstfilingtype"].value_counts(dropna=False).items():
        log.append(f"  Type {ft:>3}: {cnt:>10,} rows")

    # 5. FFIEC 031 filers only
    print("\n[5] Keeping FFIEC 031 filers only")
    df = df[df["rssdfininstfilingtype"] == config.FILING_TYPE].copy()
    _log(log, "5. Keep FFIEC 031 filers only", df)

    # 6. Minimum asset size
    print(f"\n[6] Minimum asset filter (>= ${config.MIN_ASSETS_USD:,}k)")
    df = df[df["rcfd2170"] >= config.MIN_ASSETS_USD].copy()
    _log(log, f"6. Assets >= $300m ({config.MIN_ASSETS_USD:,}k)", df)

    # 7. Drop missing total assets
    print("\n[7] Dropping rows with missing total assets")
    before = len(df)
    df = df[df["rcfd2170"].notna()].copy()
    _log(log, f"7. Drop missing rcfd2170 (removed {before - len(df):,})", df)

    # 8. Merger quarters
    print(f"\n[8] Flagging merger quarters (>{config.MERGER_JUMP_THRESHOLD:.0%} one-quarter asset jump)")
    df = df.sort_values(["rssd9001", "rssd9999"]).reset_index(drop=True)
    df["assets_lag"]   = df.groupby("rssd9001")["rcfd2170"].shift(1)
    df["asset_growth"] = (df["rcfd2170"] - df["assets_lag"]) / df["assets_lag"]
    df["merger_flag"]  = (df["asset_growth"] > config.MERGER_JUMP_THRESHOLD) & df["assets_lag"].notna()
    n_flagged = df["merger_flag"].sum()
    print(f"  Merger quarters flagged: {n_flagged:,}")
    log.append(f"  Merger quarters flagged and dropped: {n_flagged:,}")
    df = df[~df["merger_flag"]].copy()
    _log(log, "8. Drop merger quarters", df)

    # 2005Q3 dip -- confirmed against the final panel (not an earlier
    # pipeline stage, which gives different, misleading numbers). See
    # 00_config.py for the CDR/Katrina explanation.
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

    # 9. Zero-fill Schedule RC-L items
    print("\n[9] Zero-filling Schedule RC-L items (031 filers all report these)")
    for col in config.RC_L_ZERO_FILL_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            if n > 0:
                print(f"  {col}: {n:,} NaN -> 0")
                log.append(f"  {col}: {n:,} NaN -> 0 (RC-L zero-fill)")

    # rcfd3818 and NDFI-breakdown items (rcfdj454, pv-series) are NOT
    # zero-filled -- different reporting rules apply (rcfd3818 retired
    # 2010, pv-series only required 2024Q4+), so missing there doesn't
    # mean "zero".

    # 10. Final column selection & save
    print("\n[10] Final column selection")
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
    print(f"  Columns: {list(df.columns)}")

    out_path = config.F_PANEL_RAW
    df.to_parquet(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    # Coverage summary
    log += ["", "-" * 65, "FINAL VARIABLE COVERAGE:", ""]
    for col in ["rcfd3819", "rcfd3433", "rcfdj458", "rconj454", "rcfd2170", "rcfd3210",
                "rcfd1763", "rcfd1764", "rcfdj454"]:
        if col in df.columns:
            nn  = df[col].notna().sum()
            pct = nn / len(df) * 100
            log.append(f"  {col:<20} {pct:>5.1f}%  ({nn:,} / {len(df):,})")

    log_text = "\n".join(log)
    print("\n" + log_text)
    (config.OUTPUT / "cleaning_log.txt").write_text(log_text, encoding="utf-8")
    print(f"\n  Log -> {config.OUTPUT / 'cleaning_log.txt'}")


if __name__ == "__main__":
    main()