"""
08_dealscan_lender_match.py

Matches DealScan Lender_Parent_Name/Lender_Parent_Id entities to RSSD bank IDs in the Call Report panel, so DealScan loan-level records can
later be reattached to L_it. Name-matching only - produces a candidate file for manual review which I've done (was a long day), does not finalize matches.

INPUT   data/raw/dealscan_lenders.csv     (Lender_Parent_Name, Lender_Parent_Id
                                            -- distinct list, from WRDS DealScan)
        data/clean/panel_clean.csv         (rssd9001, rssd9017 -- from 02_linkage_index.py)
OUTPUT  outputs/table_dealscan_match_candidates.csv   (all candidates, for review)
        outputs/table_dealscan_match_summary.txt      (match-rate summary)
        outputs/dealscan_match_log.txt
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
SCORE_THRESHOLD_FOR_OUTPUT = 60


def _normalize_name(name: str) -> str:
    """Uppercase, strip common legal-entity suffixes at word boundaries
    only (not mid-string), collapse whitespace/punctuation. Does NOT
    strip BANK, TRUST, FINANCIAL, NATIONAL, ASSOCIATION -- these carry
    real distinguishing meaning, and token-level fuzzy matching already
    handles reordering/subsets of these words."""
    if pd.isna(name):
        return ""
    s = str(name).upper()
    s = re.sub(r"[.,]", "", s)
    strip_words = ["THE", "INC", "CORP", "CORPORATION", "CO", "LLC", "LP",
                   "N A", "NA", "LTD"]
    tokens = s.split()
    tokens = [t for t in tokens if t not in strip_words]
    s = " ".join(tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    log = ["DEALSCAN LENDER MATCHING AUDIT LOG", "=" * 65, ""]

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
    log.append(f"DealScan lenders: {len(ds):,} distinct entities")

    panel = pd.read_csv(config.CLEAN / "panel_clean.csv")
    banks = panel[["rssd9001", "rssd9017"]].drop_duplicates(subset=["rssd9001"]).reset_index(drop=True)
    banks["norm_name"] = banks["rssd9017"].apply(_normalize_name)
    log.append(f"Panel banks: {len(banks)} distinct RSSD IDs")
    bank_choices = dict(zip(banks["norm_name"], zip(banks["rssd9001"], banks["rssd9017"])))
    choice_list = list(bank_choices.keys())

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
                "confirmed_rssd9001": "",  # filled in manually during review
            })
        if has_candidate_above_threshold:
            n_any_candidate += 1

    match_df = pd.DataFrame(rows).sort_values(
        ["dealscan_name", "score"], ascending=[True, False]
    ).reset_index(drop=True)
    match_df.to_csv(config.OUTPUT / "table_dealscan_match_candidates.csv", index=False)

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
    (config.OUTPUT / "table_dealscan_match_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    (config.OUTPUT / "dealscan_match_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()