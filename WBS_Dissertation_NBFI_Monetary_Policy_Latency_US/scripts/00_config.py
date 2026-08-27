"""
Central configuration for the dissertation pipeline. All other scripts
import from this module (via importlib, since filenames start with
digits) rather than hardcoding paths or thresholds.

See README.md for more extensive notes/reasonings/acknowledgments.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUTPUT = ROOT / "outputs"
for d in [RAW, CLEAN, OUTPUT]:
    d.mkdir(parents=True, exist_ok=True)

F_RCFD1 = RAW / "rcfd1.csv"   # rcfd3820, rcfd1764, rcfd1763, rcfd3433, rcfd3819,rcfdj454, pv-series NDFI breakdown (2024Q4+ only)
F_RCFD2 = RAW / "rcfd2.csv"   # rcfdj457/458/459, rcfd2170, rcfd3210, rcfd3818, rcfd3817
F_RCONJ454 = RAW / "rconj454.csv"  # domestic-offices counterpart of rcfdj454, 2010Q1+

F_NFCI = RAW / "NFCI.csv"
F_NFCICREDIT = RAW / "NFCICREDIT.csv"
F_DGS10 = RAW / "DGS10.csv"
F_FEDFUNDS = RAW / "FEDFUNDS.csv"

# Jarocinski-Karadi shocks: https://marekjarocinski.github.io/jkshocks/jkshocks.html
# MP_pm is the headline series; CBI_pm is the companion central-bank-information shock, used as a check for CBI leakage into MP_pm.
F_SHOCKS_MONTHLY = RAW / "shocks_fed_jk_m.csv"

F_DEALSCAN_FULL_PULL = RAW / "dealscan_full_pull.csv"
F_DEALSCAN_CROSSWALK = RAW / "dealscan_confirmed_crosswalk.csv"

F_BDC_CAPIQ = RAW / "CapitalIQ_BDCs.xlsx"
F_BDC_XBRL = RAW / "Master_BDC_Layer3_Breakouts.csv"

F_PANEL_RAW = CLEAN / "panel_raw.parquet"
F_PANEL_CLEAN = CLEAN / "panel_clean.parquet"
F_PANEL_CSV = CLEAN / "panel_clean.csv"
F_PANEL_MACRO = CLEAN / "panel_macro.parquet"
F_PANEL_MACRO_CSV = CLEAN / "panel_macro.csv"
F_MACRO_QUARTERLY = CLEAN / "macro_quarterly.csv"

FILING_TYPE = 31                # FFIEC 031: foreign and domestic offices
START_DATE = "2004-10-01"       # one quarter before 2005Q1 JIC
END_DATE = "2024-09-30"         # RCFDJ458 retired 2024Q4
MIN_ASSETS_USD = 300_000        # WRDS units in thousands of dollars
MERGER_JUMP_THRESHOLD = 0.50    # one-quarter asset jump flagged as merger above this

WINSOR_LOWER = 0.01
WINSOR_HIGH = 0.99

ID_COLS = ["rssd9001", "rssd9999"]

RC_L_ZERO_FILL_COLS = [
    "rcfd3819", "rcfd3433", "rcfd3820",
    "rcfdj457", "rcfdj458", "rcfdj459", "rcfd3817",
    "rconj454",
]


# *Notes* #
# L_it = (rcfd3819 + rcfd3433 + rcfdj458 + rconj454) / rcfd2170
#   - rcfd3819  financial standby letters of credit
#   - rcfd3433  securities lent with cash collateral
#   - rcfdj458  unused commitments to FIs [2010-2024; zero before]
#   - rconj454  loans to nondepository fin. inst. [2010-2024; zero before]
#   - rcfd2170  total assets (denominator)
# rcfd3818 excluded deliberately 0 see README.