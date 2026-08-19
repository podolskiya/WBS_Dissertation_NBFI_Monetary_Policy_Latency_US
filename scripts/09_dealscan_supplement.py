"""
09_dealscan_supplement.py

Descriptive triangulation using loan-level DealScan data for the
confirmed-match subset of the panel. Not a new identified regression --
see README.md for the Khwaja-Mian scoping rationale, the Lender_Share
dollar-exposure correction, the Tranche_Active_Date vs Deal_Active_Date
choice, and the Lead/Arranger vs Participant role split.

INPUT   data/raw/dealscan_full_pull.csv     (full LPC Connector Feed pull,
                                              already on hand from the
                                              lender-matching exercise)
        data/raw/dealscan_confirmed_crosswalk.csv  (68 confirmed matches)
        data/clean/panel_macro.parquet      (L_it, credit growth, mp_shock)
OUTPUT  outputs/table_dealscan_coverage.txt
        outputs/table_dealscan_loan_summary.csv
        outputs/table_dealscan_loan_summary_by_role.csv
        outputs/fig_dealscan_spread_vs_Lit.png
        outputs/fig_dealscan_volume_vs_shock.png
        outputs/dealscan_supplement_log.txt
"""
import sys
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

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
NAVY, BLUE, RED, GRAY = "#003366", "#0066cc", "#cc0000", "#666666"
START_DATE, END_DATE = "2010-01-01", "2025-12-31"


def main():
    log = ["DEALSCAN SUPPLEMENT AUDIT LOG", "=" * 65, ""]

    crosswalk = pd.read_csv(config.F_DEALSCAN_CROSSWALK)
    crosswalk = crosswalk.dropna(subset=["dealscan_parent_id"])
    confirmed_ids = set(crosswalk["dealscan_parent_id"].astype(int))
    id_to_rssd = dict(zip(crosswalk["dealscan_parent_id"].astype(int), crosswalk["candidate_rssd9001"]))
    log.append(f"Confirmed crosswalk: {len(confirmed_ids)} lender IDs, "
               f"{crosswalk['candidate_rssd9001'].nunique()} distinct banks")

    ds = pd.read_csv(config.F_DEALSCAN_FULL_PULL, low_memory=False)
    n_total = len(ds)
    ds = ds[ds["Lender_Parent_Id"].isin(confirmed_ids)].copy()
    log.append(f"Full pull: {n_total:,} rows -> {len(ds):,} after lender filter")

    ds["rssd9001"] = ds["Lender_Parent_Id"].map(id_to_rssd)
    ds["tranche_date"] = pd.to_datetime(ds["Tranche_Active_Date"], errors="coerce")
    n_before_date = len(ds)
    ds = ds[(ds["tranche_date"] >= START_DATE) & (ds["tranche_date"] <= END_DATE)].copy()
    log.append(f"After date filter: {len(ds):,} rows (dropped {n_before_date - len(ds):,})")

    n_banks_with_loans = ds["rssd9001"].nunique()
    n_tranches = ds["LPC_Tranche_ID"].nunique() if "LPC_Tranche_ID" in ds.columns else len(ds)
    coverage_lines = [
        "TABLE: DealScan Coverage of Confirmed Panel Banks", "=" * 60, "",
        f"Confirmed banks in crosswalk: {crosswalk['candidate_rssd9001'].nunique()}",
        f"Banks with >=1 loan record in window: {n_banks_with_loans}",
        f"Total loan tranche records ({START_DATE} to {END_DATE}): {len(ds):,} "
        f"({n_tranches:,} distinct tranches)",
        "",
        "Role distribution:",
    ]
    if "Primary_Role" in ds.columns:
        for role, cnt in ds["Primary_Role"].value_counts().head(10).items():
            coverage_lines.append(f"  {role:<30} {cnt:,}")
    (config.OUTPUT / "table_dealscan_coverage.txt").write_text("\n".join(coverage_lines), encoding="utf-8")

    if n_banks_with_loans == 0:
        print("No matched loans in window -- check Lender_Parent_Id dtype/values against the crosswalk.")
        (config.OUTPUT / "dealscan_supplement_log.txt").write_text("\n".join(log), encoding="utf-8")
        return

    ds["quarter_end"] = ds["tranche_date"].dt.to_period("Q").dt.to_timestamp("Q")
    ds["facility_amount"] = pd.to_numeric(ds["Tranche_Amount_Converted"], errors="coerce")
    ds["spread"] = pd.to_numeric(ds["All_In_Spread_Drawn_bps"], errors="coerce")

    # Lender_Share is a percent of the facility -- see README for why the
    # raw facility_amount can't be used directly.
    ds["lender_share_pct"] = pd.to_numeric(ds["Lender_Share"], errors="coerce")
    n_missing_share = ds["lender_share_pct"].isna().sum()
    log.append(f"Lender_Share missing/non-numeric: {n_missing_share:,} of {len(ds):,} rows")
    ds["bank_dollar_amount"] = ds["facility_amount"] * ds["lender_share_pct"] / 100

    lead_roles = ["Admin agent", "Arranger", "Mandated Lead arranger", "Lead arranger",
                 "Syndication agent", "Bookrunner", "Documentation", "Managing agent", "Co-agent"]
    ds["role_group"] = np.where(ds["Primary_Role"].isin(lead_roles), "Lead/Arranger-type", "Participant")
    log.append(f"Role group counts: {ds['role_group'].value_counts().to_dict()}")

    bank_q = (ds.groupby(["rssd9001", "quarter_end"])
                .agg(n_loans=("bank_dollar_amount", "count"),
                     total_amount=("bank_dollar_amount", "sum"),
                     mean_spread=("spread", "mean"))
                .reset_index())
    bank_q_by_role = (ds.groupby(["rssd9001", "quarter_end", "role_group"])
                        .agg(n_loans=("bank_dollar_amount", "count"),
                             total_amount=("bank_dollar_amount", "sum"),
                             mean_spread=("spread", "mean"))
                        .reset_index())
    bank_q.to_csv(config.OUTPUT / "table_dealscan_loan_summary.csv", index=False, float_format="%.4f")
    bank_q_by_role.to_csv(config.OUTPUT / "table_dealscan_loan_summary_by_role.csv", index=False, float_format="%.4f")

    panel = pd.read_parquet(config.F_PANEL_MACRO)
    panel["rssd9999"] = pd.to_datetime(panel["rssd9999"])
    merged = bank_q.merge(
        panel[["rssd9001", "rssd9999", "L_it", "credit_growth_4q", "mp_shock", "nfci"]],
        left_on=["rssd9001", "quarter_end"], right_on=["rssd9001", "rssd9999"], how="inner"
    )
    log.append(f"Merged bank-quarter obs (DealScan x Call Report panel): {len(merged):,}")

    if len(merged) > 10:
        corr_spread_Lit = merged[["mean_spread", "L_it"]].corr().iloc[0, 1]
        corr_amount_Lit = merged[["total_amount", "L_it"]].corr().iloc[0, 1]
        log.append(f"Correlation (mean_spread, L_it): {corr_spread_Lit:.4f}")
        log.append(f"Correlation (total_amount [corrected for lender share], L_it): {corr_amount_Lit:.4f}")

        for rg in ["Lead/Arranger-type", "Participant"]:
            sub_role = bank_q_by_role[bank_q_by_role["role_group"] == rg]
            merged_role = sub_role.merge(
                panel[["rssd9001", "rssd9999", "L_it"]],
                left_on=["rssd9001", "quarter_end"], right_on=["rssd9001", "rssd9999"], how="inner"
            )
            if len(merged_role) > 10:
                c_spread = merged_role[["mean_spread", "L_it"]].corr().iloc[0, 1]
                c_amount = merged_role[["total_amount", "L_it"]].corr().iloc[0, 1]
                log.append(f"  {rg}: N={len(merged_role):,}, corr(spread,L_it)={c_spread:+.4f}, "
                          f"corr(amount,L_it)={c_amount:+.4f}")

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(merged["L_it"], merged["mean_spread"], alpha=0.4, color=NAVY, s=20)
        ax.set_xlabel("L_it (bank-quarter)")
        ax.set_ylabel("Mean loan spread (bps, DealScan)")
        ax.set_title("DealScan loan pricing vs. bank-NBFI linkage intensity\n"
                     "(descriptive triangulation, confirmed-match banks only)")
        fig.tight_layout()
        fig.savefig(config.OUTPUT / "fig_dealscan_spread_vs_Lit.png", bbox_inches="tight")
        plt.close()

        agg_q = merged.groupby("quarter_end").agg(
            total_amount=("total_amount", "sum"), mp_shock=("mp_shock", "mean")
        ).reset_index()
        fig, ax1 = plt.subplots(figsize=(10, 5.5))
        ax1.bar(agg_q["quarter_end"], agg_q["total_amount"], width=60, color=NAVY, alpha=0.7,
                label="Aggregate loan volume (confirmed-match banks)")
        ax1.set_ylabel("Aggregate loan volume ($mm)", color=NAVY)
        ax2 = ax1.twinx()
        ax2.plot(agg_q["quarter_end"], agg_q["mp_shock"], color=RED, lw=1.5, label="mp_shock")
        ax2.set_ylabel("mp_shock", color=RED)
        ax1.set_title("DealScan loan volume vs. monetary policy shock\n(confirmed-match banks)")
        fig.tight_layout()
        fig.savefig(config.OUTPUT / "fig_dealscan_volume_vs_shock.png", bbox_inches="tight")
        plt.close()

    (config.OUTPUT / "dealscan_supplement_log.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()