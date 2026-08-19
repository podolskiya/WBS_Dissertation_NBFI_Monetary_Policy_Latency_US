"""
08_dealscan_lender_match.py
----------------------------
PURPOSE
    Match DealScan's Lender_Parent_Name/Lender_Parent_Id entities to
    the RSSD bank IDs in the Call Report panel, so DealScan loan-level
    records can eventually be joined to L_it. This is a NAME-MATCHING
    step only -- it produces a candidate-match review file, it does
    NOT automatically finalize matches. Bank name matching always
    needs a human to confirm the candidates (fuzzy string similarity
    alone will produce both false positives -- similarly-named but
    different institutions -- and false negatives -- genuinely the
    same institution under very different naming conventions).

    KNOWN COMPLICATION (see conversation notes): DealScan's
    Lender_Parent_Name reflects the CURRENT corporate structure
    applied retroactively to historical loans (e.g. "Truist Financial"
    appears as the parent for SunTrust Bank loans from 2010-2012,
    years before Truist existed as a merger of BB&T and SunTrust in
    Dec 2019). This script only produces a NAME match; a separate,
    later step is needed to build a TIME-AWARE crosswalk mapping each
    DealScan parent to the RSSD ID that actually existed at each
    loan's date, once we know which banks match at all.

INPUT   data/raw/dealscan_lenders.csv     (Lender_Parent_Name, Lender_Parent_Id
                                            -- distinct list, from WRDS DealScan)
        data/clean/panel_clean.csv         (rssd9001, rssd9017 -- from 02_linkage_index.py)

OUTPUT  outputs/table_dealscan_match_candidates.csv   (all candidates, for review)
        outputs/table_dealscan_match_summary.txt      (match-rate summary)
        outputs/dealscan_match_log.txt

MATCHING METHOD
    Normalizes both name sets (uppercase, strips common legal-entity
    suffixes: INC, CORP, CO, LLC, LP, N.A., NA, THE, at start/end only
    -- not mid-string, to avoid corrupting names where these appear as
    real words). Uses rapidfuzz token_sort_ratio (handles word-order
    differences, e.g. "Bank of Hawaii" vs "Hawaii Bank") to score every
    DealScan lender against every panel bank, keeping the top 3
    candidates per lender for manual review.

REVIEW WORKFLOW
    Open table_dealscan_match_candidates.csv. For each dealscan_name,
    confirm (or reject) the top candidate:
    - score >= 90: usually a safe accept, but still eyeball it
    - score 75-89: needs a real look (common for legal-suffix
      differences, abbreviations, or genuinely different banks with
      similar names)
    - score < 75: likely no match in your 194-bank sample (most
      DealScan lenders are NOT in your Call Report panel -- large
      non-bank lenders, foreign banks, insurers, asset managers, etc.
      are expected and should be excluded, not force-matched)
    Add a "confirmed_rssd9001" column as you review; send the reviewed
    file back and the next script will use only confirmed rows.
"""

import sys
import re
import warnings
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
config = import_module("00_config")

warnings.filterwarnings("ignore")

TOP_N_CANDIDATES = 3
SCORE_THRESHOLD_FOR_OUTPUT = 60  # don't bother outputting candidates below this


def _normalize_name(name: str) -> str:
    """
    Uppercase, strip common legal-entity suffixes/prefixes (at word
    boundaries only, not mid-string), collapse whitespace/punctuation.
    Deliberately does NOT strip words like BANK, TRUST, FINANCIAL,
    NATIONAL, ASSOCIATION -- these carry real distinguishing meaning
    for many institutions and token-level fuzzy matching handles
    reordering/subsets of these words already.
    """
    if pd.isna(name):
        return ""
    s = str(name).upper()
    s = re.sub(r"[.,]", "", s)
    # strip common corporate suffixes/prefixes as whole words only
    strip_words = ["THE", "INC", "CORP", "CORPORATION", "CO", "LLC", "LP",
                   "N A", "NA", "LTD"]
    tokens = s.split()
    tokens = [t for t in tokens if t not in strip_words]
    s = " ".join(tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    print("\n" + "=" * 65)
    print("08_dealscan_lender_match.py  --  DealScan lender to RSSD matching")
    print("=" * 65)

    log = ["DEALSCAN LENDER MATCHING AUDIT LOG", "=" * 65, ""]

    # ── [1] Load DealScan lenders ────────────────────────────────────────
    print("\n[1] Loading DealScan distinct lender list")
    ds_path = config.RAW / "dealscan_full_pull.csv"
    if not ds_path.exists():
        # fall back to the raw pasted-text format used during development/testing
        ds_path = config.RAW / "dealscan_full_pull.txt"
        ds = pd.read_csv(ds_path, sep="\t", header=None,
                         names=["Lender_Parent_Name", "Lender_Parent_Id"])
    else:
        ds = pd.read_csv(ds_path)
    ds = ds.drop_duplicates(subset=["Lender_Parent_Name"]).reset_index(drop=True)
    ds["norm_name"] = ds["Lender_Parent_Name"].apply(_normalize_name)
    print(f"  {len(ds):,} distinct DealScan lender_parent entities")
    log.append(f"DealScan lenders: {len(ds):,} distinct entities")

    # ── [2] Load panel bank names ────────────────────────────────────────
    print("\n[2] Loading Call Report panel bank names")
    panel = pd.read_csv(config.CLEAN / "panel_clean.csv")
    banks = panel[["rssd9001", "rssd9017"]].drop_duplicates(subset=["rssd9001"]).reset_index(drop=True)
    banks["norm_name"] = banks["rssd9017"].apply(_normalize_name)
    print(f"  {len(banks)} distinct banks in panel")
    log.append(f"Panel banks: {len(banks)} distinct RSSD IDs")

    bank_choices = dict(zip(banks["norm_name"], zip(banks["rssd9001"], banks["rssd9017"])))
    choice_list = list(bank_choices.keys())

    # ── [3] Fuzzy match each DealScan lender against the bank list ──────
    print(f"\n[3] Matching (top {TOP_N_CANDIDATES} candidates per lender, "
          f"rapidfuzz token_sort_ratio)")
    rows = []
    n_any_candidate = 0
    for _, row in ds.iterrows():
        if not row["norm_name"]:
            continue
        matches = process.extract(
            row["norm_name"], choice_list, scorer=fuzz.token_sort_ratio,
            limit=TOP_N_CANDIDATES
        )
        has_candidate_above_threshold = False
        for matched_name, score, _ in matches:
            if score < SCORE_THRESHOLD_FOR_OUTPUT:
                continue
            has_candidate_above_threshold = True
            rssd, orig_bank_name = bank_choices[matched_name]
            rows.append({
                "dealscan_name": row["Lender_Parent_Name"],
                "dealscan_parent_id": row["Lender_Parent_Id"],
                "candidate_rssd9001": rssd,
                "candidate_bank_name": orig_bank_name,
                "score": score,
                "confirmed_rssd9001": "",  # for manual fill-in during review
            })
        if has_candidate_above_threshold:
            n_any_candidate += 1

    match_df = pd.DataFrame(rows).sort_values(
        ["dealscan_name", "score"], ascending=[True, False]
    ).reset_index(drop=True)
    match_df.to_csv(config.OUTPUT / "table_dealscan_match_candidates.csv", index=False)
    print(f"  {len(ds)} DealScan lenders checked, {n_any_candidate} have >=1 "
          f"candidate above score {SCORE_THRESHOLD_FOR_OUTPUT}")
    print(f"  Saved -> {config.OUTPUT / 'table_dealscan_match_candidates.csv'}")

    # ── [4] Summary ───────────────────────────────────────────────────────
    print("\n[4] Match quality summary")
    high_conf = match_df[match_df["score"] >= 90]["dealscan_name"].nunique()
    med_conf = match_df[(match_df["score"] >= 75) & (match_df["score"] < 90)]["dealscan_name"].nunique()
    low_conf = match_df[(match_df["score"] >= SCORE_THRESHOLD_FOR_OUTPUT) & (match_df["score"] < 75)]["dealscan_name"].nunique()
    no_match = len(ds) - n_any_candidate

    summary_lines = [
        "TABLE: DealScan Lender Matching Summary", "=" * 60, "",
        f"Total DealScan lender_parent entities checked: {len(ds):,}",
        f"  High confidence (score >= 90):        {high_conf:,} lenders",
        f"  Needs review (score 75-89):            {med_conf:,} lenders",
        f"  Low confidence (score {SCORE_THRESHOLD_FOR_OUTPUT}-74):          {low_conf:,} lenders",
        f"  No candidate above {SCORE_THRESHOLD_FOR_OUTPUT}:              {no_match:,} lenders "
        f"(expected -- most DealScan lenders are not in your 194-bank sample)",
        "",
        "NEXT STEP: open table_dealscan_match_candidates.csv, fill in",
        "'confirmed_rssd9001' for every row you accept as a genuine match,",
        "and send the reviewed file back. Remember: a name match here does",
        "NOT yet account for merger timing (see script docstring) -- that is",
        "a separate follow-up step once we know the confirmed match set.",
    ]
    summary_text = "\n".join(summary_lines)
    (config.OUTPUT / "table_dealscan_match_summary.txt").write_text(summary_text, encoding="utf-8")
    print(f"  Saved -> {config.OUTPUT / 'table_dealscan_match_summary.txt'}")
    print("\n" + summary_text)

    log_text = "\n".join(log)
    (config.OUTPUT / "dealscan_match_log.txt").write_text(log_text, encoding="utf-8")
    print(f"\n  Log -> {config.OUTPUT / 'dealscan_match_log.txt'}")
    print("\n  Done.")


if __name__ == "__main__":
    main()