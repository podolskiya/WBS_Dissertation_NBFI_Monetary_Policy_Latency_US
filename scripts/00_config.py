"""
00_config.py
------------
Central config for the dissertation pipeline. Every other script imports
from here (via importlib, since the filenames start with digits) rather
than hardcoding paths or thresholds, so a change only has to happen once.

Quick history, for my own sanity as much as anyone else's:
  - Originally built around a 4-file WRDS pull (rcfd_old.csv + 3 patch
    files). That legacy file is gone now -- the consolidated 2-file pull
    (rcfd1.csv + rcfd2.csv) covers every core L_it component for FFIEC 031
    filers from 2004Q4 onward, which is the whole window I need.
  - RCFD1520 (drawn loans to non-depository FIs -- the item I actually
    wanted at the start of this) is not available under the WRDS
    subscription tier I have (Bank Regulatory Premium). RCFDJ454 looked
    like a possible substitute but turned out to be zero-populated for
    almost the entire sample (only starts 2024Q4, and only for the
    largest banks) -- not usable. See the L_it notes below for how this
    ended up getting resolved anyway, via RCONJ454.
  - Window originally ran through 2025Q4. Had to trim it back to 2024Q3
    after finding FFIEC actually retired one of the RC-L items the index
    depends on starting 2024Q4 (see END_DATE below) -- not fixable by
    re-pulling, the item just doesn't exist anymore in that form.
"""
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
RAW    = ROOT / "data" / "raw"
CLEAN  = ROOT / "data" / "clean"
OUTPUT = ROOT / "outputs"
for d in [RAW, CLEAN, OUTPUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Raw file names ───────────────────────────────────────────────────────
# The consolidated 2-file WRDS pull. Splitting fields across two files
# wasn't my choice -- it's just how the WRDS query builder capped the
# column count per request.
F_RCFD1 = RAW / "rcfd1.csv"   # rcfd3820, rcfd1764, rcfd1763, rcfd3433, rcfd3819,
                               # rcfdj454, full pv-series (NDFI breakdown, 2024Q4+ only)
F_RCFD2 = RAW / "rcfd2.csv"   # rcfdj457/458/459, rcfd2170, rcfd3210, rcfd3818, rcfd3817

# RCONJ454 -- pulled as a separate, later addition once I realised what it
# was. Domestic-offices ("RCON") counterpart of RCFDJ454 (which is
# consolidated worldwide, "RCFD"). For basically every bank in this
# $300m+ sample, none of which have foreign offices, RCON and RCFD are
# numerically identical -- so this isn't a compromise, it's the same
# number under a different reporting basis. RCFDJ454 in the main pull
# only starts populating 2024Q4+, which I initially assumed meant the
# item just didn't exist before then. It does -- WRDS's own query for
# THAT specific field just didn't reach back further. RCONJ454, same
# underlying concept, has continuous coverage from 2010Q1. This is what
# ended up resolving the RCFD1520 problem, at least for 2010 onward,
# essentially by accident.
F_RCONJ454 = RAW / "rconj454.csv"

# ── FRED macro series (downloaded manually -- no API used, see
# 03_macro_merge.py for the messy CSV-encoding details) ──────────────────
F_NFCI       = RAW / "NFCI.csv"
F_NFCICREDIT = RAW / "NFCICREDIT.csv"
F_DGS10      = RAW / "DGS10.csv"
F_FEDFUNDS   = RAW / "FEDFUNDS.csv"

# ── Monetary policy shock series (Jarocinski-Karadi) ─────────────────────
# Source: https://marekjarocinski.github.io/jkshocks/jkshocks.html
# Author's site says "updated until July 2025". MP_pm is the primary
# series (poor-man's sign-restriction identification); CBI_pm is the
# companion central-bank-information shock, useful as a check for
# whether the "MP shock" is actually just central bank information
# leaking through. MP_median/CBI_median are the median-rotation variant,
# kept around for robustness but not used as the headline series.
F_SHOCKS_MONTHLY = RAW / "shocks_fed_jk_m.csv"

# ── DealScan supplement ───────────────────────────────────────────────────
# Full LPC Connector Feed pull -- every lender_parent entity, not
# filtered down yet. Filtering to the confirmed matches happens later.
F_DEALSCAN_FULL_PULL = RAW / "dealscan_full_pull.csv"
# The lender_parent_id -> RSSD crosswalk, built by hand-reviewing fuzzy
# name matches (08_dealscan_lender_match.py). A handful of entries got
# dropped on purpose during that review -- a few structurally ambiguous
# parent entities (e.g. one DealScan parent covering two legally
# separate subsidiary banks) and two merger-timing cases where the
# current entity name doesn't reliably map to one RSSD across the whole
# sample window. Documented in the review notes, not just silently
# excluded.
F_DEALSCAN_CROSSWALK = RAW / "dealscan_confirmed_crosswalk.csv"

# ── BDC supplementary data ────────────────────────────────────────────────
# Primary source: Capital IQ, pulled across the whole BDC industry
# classification (67 tickers resolved, quarterly, 2010-2025).
F_BDC_CAPIQ = RAW / "CapitalIQ_BDCs.xlsx"
# Cross-check against SEC XBRL company-facts data (38 tickers, patchier
# coverage). Only using the LineOfCreditFacilityMaximumBorrowingCapacity
# tag from this -- it's the one field in the XBRL pull that unambiguously
# means "bank credit facility capacity." The other tags in that file
# (DebtInstrumentCarryingAmount etc.) are too generic to trust without
# more work than this cross-check is worth.
F_BDC_XBRL = RAW / "Master_BDC_Layer3_Breakouts.csv"

# ── Cleaned output files ──────────────────────────────────────────────────
F_PANEL_RAW   = CLEAN / "panel_raw.parquet"      # after 01_data_cleaning.py
F_PANEL_CLEAN = CLEAN / "panel_clean.parquet"    # after 02_linkage_index.py
F_PANEL_CSV   = CLEAN / "panel_clean.csv"
F_PANEL_MACRO      = CLEAN / "panel_macro.parquet"  # after 03_macro_merge.py
F_PANEL_MACRO_CSV  = CLEAN / "panel_macro.csv"
F_MACRO_QUARTERLY  = CLEAN / "macro_quarterly.csv"  # just for eyeballing the macro series

# ── Sample definition ──────────────────────────────────────────────────────
FILING_TYPE    = 31             # FFIEC 031 (banks with foreign AND domestic offices)
START_DATE     = "2004-10-01"   # one quarter before 2005Q1, so the first real
                                 # observation has a valid lag
END_DATE       = "2024-09-30"   # Originally had this at 2025-12-31. Had to walk it
                                 # back after finding (the hard way -- see
                                 # 01_data_cleaning.py notes on 2024Q4+) that FFIEC
                                 # discontinued RCFDJ458 ("unused commitments to
                                 # FIs") starting the 2024Q4 filing cycle. Checked
                                 # this directly against the FFIEC bulk Call Report
                                 # schedule for Dec 2024 -- the column simply isn't
                                 # in the RC-L schedule anymore. RCFDJ457/459 are
                                 # still there but got repurposed to mean something
                                 # different (drawn loan balances, not unused
                                 # commitments), so they can't substitute either.
                                 # This is a genuine item retirement, not a pull
                                 # configuration problem like the RCFD1520 one was --
                                 # there's no re-pull that fixes it. Panel ends where
                                 # the underlying item stops existing.
MIN_ASSETS_USD = 300_000        # $300m, in WRDS's thousands-of-dollars units
MERGER_JUMP_THRESHOLD = 0.50    # flag any one-quarter asset jump bigger than this
                                 # as a likely merger, not organic growth

# ── Winsorisation (applied in 02_linkage_index.py) ────────────────────────
WINSOR_LOWER = 0.01
WINSOR_HIGH  = 0.99

# ── Identifier columns ──────────────────────────────────────────────────────
ID_COLS = ["rssd9001", "rssd9999"]

# ── Off-balance-sheet items to zero-fill when missing ────────────────────
# FFIEC 031 filers complete Schedule RC-L in full, so a missing value
# from a 031 filer means "genuinely zero / not applicable," not "not
# reported." This is NOT the same as a missing rcfd2170 (total assets),
# which gets dropped outright in 01_data_cleaning.py rather than
# zero-filled -- a bank can plausibly have zero standby letters of
# credit, it can't plausibly have zero total assets.
RC_L_ZERO_FILL_COLS = [
    "rcfd3819", "rcfd3433", "rcfd3820",
    "rcfdj457", "rcfdj458", "rcfdj459", "rcfd3817",
    "rconj454",  # same zero-fill logic as rcfdj458 -- the item genuinely
                 # didn't exist before 2010, so treating it as zero for
                 # that period is a defensible modelling choice, not a
                 # data-quality patch
]

# ── L_it construction ──────────────────────────────────────────────────────
#
# Current definition:
#   L_it = (rcfd3819 + rcfd3433 + rcfdj458 + rconj454) / rcfd2170
#
#   rcfd3819  Financial standby letters of credit   [full coverage 2005-2024]
#   rcfd3433  Securities lent with cash collateral  [full coverage 2005-2024]
#   rcfdj458  Unused commitments to FIs             [2010-2024; zero before]
#   rconj454  Loans to nondepository fin. inst.     [2010-2024; zero before]
#   rcfd2170  Total assets (denominator)
#
# On RCFD1520 (drawn loans to NDFI) -- this was the item I originally
# wanted and couldn't get (requires a WRDS subscription tier I don't
# have). RCONJ454 turned out to be the domestic-office version of the
# same underlying MDRM concept, with continuous coverage back to 2010,
# confirmed by directly comparing FFIEC's own bulk Call Report files for
# 2010Q1 and 2024Q4 side by side -- RCFD1520 wasn't in either one
# (retired well before 2010), so this isn't something WRDS was hiding
# behind a paywall, it's genuinely gone from the reporting form. This
# was a useful accident more than a plan.
#
# On rcfd3818 (unused commitments -- other, aggregate) -- deliberately
# NOT included. This is a catch-all bucket, and for a handful of FFIEC
# 031 trust-company filers it comes out larger than total assets, which
# would have quietly poisoned L_it with commitments that have nothing to
# do with NBFI exposure. FFIEC retired this item in 2010 anyway, in
# favour of the more precise rcfdj457/458/459 breakdown, so excluding it
# lines up with how the reporting form itself evolved. Kept in the
# cleaned panel for reference, just not used in the index.
#
# 2005Q3 -- known dip in bank coverage (documented properly with numbers
# and a citation in 01_data_cleaning.py, once I'd actually confirmed it
# against the final panel rather than an earlier pipeline stage). Traces
# back to two things happening at once: the CDR electronic filing system
# went live that exact quarter, and Hurricane Katrina hit in the middle
# of the same reporting period. Left alone, not patched -- there's a
# real, citable reason for it.