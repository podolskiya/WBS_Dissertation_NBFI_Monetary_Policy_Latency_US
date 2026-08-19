"""
04_regression_p1.py
--------------------
PURPOSE
    Estimate the P1 monetary policy transmission local projections
    (Jorda, 2005): does bank-NBFI linkage intensity (L_it) alter the
    strength of transmission of monetary policy shocks to bank credit
    growth, at horizons h = 0..8 quarters?

SPECIFICATION
    Outcome (h-quarter cumulative log change, relative to quarter t-1
    -- the standard Jorda local-projections construction):
        Delta_h y_{i,t} = log(CI_{i,t+h}) - log(CI_{i,t-1})
    Built via calendar-based merges (robust to panel gaps, e.g. the
    documented 2005Q3 WRDS extraction gap), not positional shifts.
    Because the outcome is already cumulative from t-1, the
    interaction coefficient at h=8 IS the "cumulative eight-quarter
    effect" the dissertation plan asks for -- no further summation
    needed. Peak effect = max|delta_h| across h; time-to-peak = the
    h at which that max occurs.

    Model A (PRIMARY) -- bank + quarter fixed effects:
        Delta_h y_{i,t} = alpha_i + gamma_t + delta_h (mp_shock_t x L_it)
                          + phi_h L_it + theta_h' X_it + u_{i,t+h}
        mp_shock_t is absorbed by quarter FE (it is constant across
        banks within a quarter) -- so beta_h (average transmission)
        is NOT separately estimable here. delta_h (the interaction) IS
        estimable, because L_it varies across banks within a quarter.
        This is the tighter identification: quarter FE absorb *any*
        common shock that quarter, not just mp_shock, so delta_h is
        not contaminated by omitted aggregate confounds. This directly
        answers the research question (does linkage alter
        transmission?) even though it cannot report an average
        transmission effect on its own.

    Model B (ROBUSTNESS) -- bank FE only, no quarter FE:
        Delta_h y_{i,t} = alpha_i + beta_h mp_shock_t + delta_h (mp_shock_t x L_it)
                          + phi_h L_it + psi_h' Z_t + theta_h' X_it + u_{i,t+h}
        Z_t = nfci, term_spread (aggregate controls, since no quarter FE
        to absorb them). Both beta_h and delta_h are estimable here,
        matching the dissertation plan's literal spec, at the cost of
        weaker identification (aggregate confounds not captured by
        Z_t contaminate both coefficients).

    Both models: two-way clustered SEs (bank, quarter) via
    linearmodels.PanelOLS.

    Model C (ROBUSTNESS) -- bank + quarter FE, LAGGED linkage:
        Same as Model A but interacts mp_shock_t with L_{i,t-1}
        instead of contemporaneous L_it. Avoids simultaneity: the
        shock itself could move L_it within the same quarter,
        contaminating the interaction. Standard practice in the
        bank-lending-channel literature (Kashyap-Stein 1995,
        Kishan-Opiela 2000).

    Model D (ROBUSTNESS) -- bank + quarter FE, TIME-AVERAGED linkage:
        Same as Model A but interacts mp_shock_t with each bank's
        full-sample average L_it (a time-invariant "linkage type")
        instead of the noisy quarterly flow measure. L_it has a large
        mass at exactly zero and moves in/out of securities lending
        quarter to quarter; using the noisy contemporaneous value can
        mechanically attenuate the interaction toward zero even if a
        real effect exists. The main effect of this variable is
        absorbed by entity (bank) FE since it is time-invariant; only
        the interaction with mp_shock_t is estimable, which is exactly
        the parameter of interest.

    Model E (EXTENSION) -- bank + quarter FE, stress-dependent triple
        interaction: does the transmission-heterogeneity effect
        (mp_shock x L_it) get amplified during high-financial-stress
        quarters (NFCI in the top historical tercile)? Motivated
        directly by the dissertation plan's Brunnermeier-Pedersen
        funding-stress discussion (already central to Proposition 2)
        and by Models A-C's unconditional null: if the effect is real
        but concentrated in stress periods, an unconditional average
        against near-zero calm-period effects would look null even
        though a genuine state-dependent effect exists.

INPUT   data/clean/panel_macro.parquet

OUTPUT  outputs/table_p1_local_projections.csv   (all horizons, all 4 models)
        outputs/table_p1_local_projections.txt   (formatted, write-up ready)
        outputs/fig_p1_impulse_response.png      (two panels: FE comparison
                                                   A-vs-B; linkage-variable
                                                   comparison A-vs-C-vs-D)
        outputs/p1_regression_log.txt

NOTE ON FREQUENCY MISMATCH (supervisor comment, carried from 03_macro_merge.py)
    mp_shock_t is a quarterly SUM of monthly-aggregated FOMC surprises.
    This regression identifies the effect of the quarterly-cumulated
    shock, not of an individual FOMC announcement -- state this
    explicitly in the methodology write-up, not just in code comments.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from linearmodels.panel import PanelOLS
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
config = import_module("00_config")

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
NAVY, BLUE, RED, GRAY = "#003366", "#0066cc", "#cc0000", "#666666"
GREEN = "#006633"

HORIZONS   = list(range(0, 9))          # h = 0..8 quarters
CONTROLS   = ["capital_ratio", "log_assets", "ci_loan_growth"]
WINSOR_LO, WINSOR_HI = 0.01, 0.99
INTERACT_VAR = "shock_x_L"


def _build_horizon_outcomes(df):
    """
    Build Delta_h y_{i,t} = log(CI_{i,t+h}) - log(CI_{i,t-1}) for
    h = 0..8 via calendar-based merges.

    Pattern (for both the t-1 base and the t+h forward level): shift
    the MAIN row's own date to the target quarter, then merge against
    an unshifted copy of (bank, date, ci_level) on that target date.
    Shifting the source copy instead of the main row's target date
    (an earlier version of this function did that) silently computes
    values from h quarters in the PAST rather than the future --
    caught via y_h1 having exactly zero variance in testing (h=0 is
    shift-symmetric so it happened to be unaffected).
    """
    df = df.sort_values(["rssd9001", "rssd9999"]).reset_index(drop=True)
    src = df[["rssd9001", "rssd9999", "ci_level"]].copy()

    # -- lag-1 base level (t-1) --
    df["rssd9999_target"] = df["rssd9999"] - pd.DateOffset(months=3)
    df["rssd9999_target"] = df["rssd9999_target"] + pd.offsets.QuarterEnd(0)
    lag_src = src.rename(columns={"rssd9999": "rssd9999_target", "ci_level": "ci_lag1"})
    df = df.merge(lag_src, on=["rssd9001", "rssd9999_target"], how="left")
    df = df.drop(columns=["rssd9999_target"])

    # -- forward levels (t+h) for each horizon --
    for h in HORIZONS:
        df["rssd9999_target"] = df["rssd9999"] + pd.DateOffset(months=3 * h)
        df["rssd9999_target"] = df["rssd9999_target"] + pd.offsets.QuarterEnd(0)
        fwd_src = src.rename(columns={"rssd9999": "rssd9999_target", "ci_level": f"ci_fwd{h}"})
        df = df.merge(fwd_src, on=["rssd9001", "rssd9999_target"], how="left")
        df = df.drop(columns=["rssd9999_target"])

        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.log(df[f"ci_fwd{h}"] / df["ci_lag1"])
        raw = raw.replace([np.inf, -np.inf], np.nan)
        n_valid = raw.notna().sum()
        if n_valid > 10:
            lo, hi = raw.quantile([WINSOR_LO, WINSOR_HI])
            df[f"y_h{h}"] = raw.clip(lower=lo, upper=hi)
        else:
            df[f"y_h{h}"] = raw

    return df


def _drop_singletons(est: pd.DataFrame, drop_time_singletons: bool = True) -> pd.DataFrame:
    """
    Drop singleton entities (bank appears only once) and, if
    drop_time_singletons, singleton time periods (quarter has only one
    bank) before fitting an FE model. A singleton is perfectly absorbed
    by its own fixed effect -- it contributes zero identifying
    information but corrupts the residual degrees-of-freedom count
    linearmodels uses internally (causes a ZeroDivisionError in fit()
    otherwise). Iterates because dropping one singleton can
    occasionally create another. Only entity singletons matter when
    there is no time-effects term (Model B).
    """
    prev_len = -1
    while len(est) != prev_len:
        prev_len = len(est)
        bank_counts = est["rssd9001"].value_counts()
        good_banks = bank_counts[bank_counts > 1].index
        est = est[est["rssd9001"].isin(good_banks)]
        if drop_time_singletons:
            q_counts = est["rssd9999"].value_counts()
            good_qs = q_counts[q_counts > 1].index
            est = est[est["rssd9999"].isin(good_qs)]
    return est


def _fit_bankquarter_fe(df, h, L_col, model_label):
    """
    Shared estimator for the bank+quarter FE specification (Model A's
    identification strategy), parameterized by which linkage variable
    is used for the main effect and the shock interaction. Used by
    Model A (contemporaneous L_it), Model C (lagged L_it, avoids
    simultaneity with the shock), and Model D (bank-level time-average
    "linkage type", avoids quarter-to-quarter measurement noise).
    """
    cols = [f"y_h{h}", L_col, "mp_shock"] + CONTROLS
    est = df[["rssd9001", "rssd9999"] + cols].dropna().copy()
    est["shock_x_L"] = est["mp_shock"] * est[L_col]
    n_before_singleton = len(est)
    est = _drop_singletons(est)
    n_singleton_dropped = n_before_singleton - len(est)
    n_banks = est["rssd9001"].nunique()
    est = est.set_index(["rssd9001", "rssd9999"])
    y = est[f"y_h{h}"]
    X = est[[L_col, INTERACT_VAR] + CONTROLS]
    try:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        n_clusters_bank = est.index.get_level_values(0).nunique()
        n_clusters_time = est.index.get_level_values(1).nunique()
        two_way_se = res.std_errors.get(INTERACT_VAR, np.nan)
        needs_fallback = pd.isna(two_way_se)
        fallback_reason = "NaN"
        if not needs_fallback:
            # Two-way cluster-robust covariance (Cameron-Gelbach-Miller) can
            # be numerically degenerate WITHOUT returning NaN -- it can yield
            # an artificially deflated SE instead, which looks like a real
            # (and spuriously significant) result unless checked directly.
            # Compare against the one-way (bank) clustered SE as a sanity
            # check: a two-way SE that is a small fraction of the one-way SE
            # is a red flag for this pathology, not a genuine, more-precise
            # estimate.
            res_oneway_check = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            one_way_se = res_oneway_check.std_errors.get(INTERACT_VAR, np.nan)
            if not pd.isna(one_way_se) and one_way_se > 0 and two_way_se < 0.2 * one_way_se:
                needs_fallback = True
                fallback_reason = (f"deflated (two-way SE={two_way_se:.4f} is <20% of "
                                   f"one-way SE={one_way_se:.4f} -- likely degenerate "
                                   f"covariance, not genuine precision)")
        if needs_fallback:
            print(f"    NOTE h={h}: two-way clustered SE is {fallback_reason} for {INTERACT_VAR} ({model_label}) "
                  f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time}) "
                  f"-- falling back to one-way (bank) clustering")
            res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            res._cov_note = (f"FALLBACK: one-way (bank) clustered SE used -- two-way "
                             f"clustered covariance was {fallback_reason} at this horizon "
                             f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time})")
        else:
            res._cov_note = None
        return res, len(est), n_banks, n_singleton_dropped
    except Exception as e:
        print(f"    {model_label} FAILED at h={h}: {e}")
        return None, len(est), n_banks, n_singleton_dropped


def _fit_model_A(df, h):
    """Bank + quarter FE, contemporaneous L_it. PRIMARY specification."""
    return _fit_bankquarter_fe(df, h, "L_it", "Model A")


def _fit_model_B(df, h):
    """Bank FE only. Both mp_shock (beta_h) and interaction (delta_h) estimable."""
    cols = [f"y_h{h}", "L_it", "mp_shock", "nfci", "term_spread"] + CONTROLS
    est = df[["rssd9001", "rssd9999"] + cols].dropna().copy()
    est["shock_x_L"] = est["mp_shock"] * est["L_it"]
    est = _drop_singletons(est, drop_time_singletons=False)
    n_banks = est["rssd9001"].nunique()
    est = est.set_index(["rssd9001", "rssd9999"])
    y = est[f"y_h{h}"]
    X = est[["mp_shock", "L_it", INTERACT_VAR, "nfci", "term_spread"] + CONTROLS]
    try:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=False, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        n_clusters_bank = est.index.get_level_values(0).nunique()
        n_clusters_time = est.index.get_level_values(1).nunique()
        two_way_se = res.std_errors.get(INTERACT_VAR, np.nan)
        needs_fallback = pd.isna(two_way_se)
        fallback_reason = "NaN"
        if not needs_fallback:
            res_oneway_check = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            one_way_se = res_oneway_check.std_errors.get(INTERACT_VAR, np.nan)
            if not pd.isna(one_way_se) and one_way_se > 0 and two_way_se < 0.2 * one_way_se:
                needs_fallback = True
                fallback_reason = (f"deflated (two-way SE={two_way_se:.4f} is <20% of "
                                   f"one-way SE={one_way_se:.4f} -- likely degenerate "
                                   f"covariance, not genuine precision)")
        if needs_fallback:
            print(f"    NOTE h={h}: two-way clustered SE is {fallback_reason} for {INTERACT_VAR} (Model B) "
                  f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time}) "
                  f"-- falling back to one-way (bank) clustering")
            res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            res._cov_note = (f"FALLBACK: one-way (bank) clustered SE used -- two-way "
                             f"clustered covariance was {fallback_reason} at this horizon "
                             f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time})")
        else:
            res._cov_note = None
        return res, len(est), n_banks
    except Exception as e:
        print(f"    Model B FAILED at h={h}: {e}")
        return None, len(est), n_banks


def _fit_model_C(df, h):
    """Bank + quarter FE, L_{i,t-1} (lagged linkage). Avoids simultaneity:
    the shock itself could move contemporaneous L_it within the same
    quarter; interacting with the pre-shock value is standard practice
    in the bank-lending-channel literature (Kashyap-Stein 1995,
    Kishan-Opiela 2000)."""
    return _fit_bankquarter_fe(df, h, "L_it_lag1", "Model C")


def _fit_model_D(df, h):
    """Bank + quarter FE, time-averaged L_it ("linkage type"). Avoids
    attenuation from quarter-to-quarter measurement noise in the flow
    measure -- L_it has a large mass at exactly zero and jumps in/out
    of securities lending, which mechanically attenuates an interaction
    coefficient toward zero if used contemporaneously."""
    return _fit_bankquarter_fe(df, h, "L_it_bankmean", "Model D")


def _build_alt_linkage_vars(df):
    """
    Build L_it_lag1 (calendar-based one-quarter lag, same corrected
    merge pattern as the horizon outcomes -- shift the MAIN row's
    target date, not the source copy) and L_it_bankmean (bank-level
    time-average, time-invariant so its main effect is absorbed by
    entity FE but the interaction with mp_shock_t remains estimable).
    """
    df = df.sort_values(["rssd9001", "rssd9999"]).reset_index(drop=True)
    src = df[["rssd9001", "rssd9999", "L_it"]].copy()
    df["rssd9999_target"] = df["rssd9999"] - pd.DateOffset(months=3)
    df["rssd9999_target"] = df["rssd9999_target"] + pd.offsets.QuarterEnd(0)
    lag_src = src.rename(columns={"rssd9999": "rssd9999_target", "L_it": "L_it_lag1"})
    df = df.merge(lag_src, on=["rssd9001", "rssd9999_target"], how="left")
    df = df.drop(columns=["rssd9999_target"])

    df["L_it_bankmean"] = df.groupby("rssd9001")["L_it"].transform("mean")
    return df


def _build_stress_indicator(df, threshold_pct=2/3):
    """
    Flag high-stress quarters as NFCI in the top tercile (default) of
    its HISTORICAL distribution. Computed on the unique quarterly NFCI
    series (each quarter counted once), not the bank-quarter panel --
    otherwise quarters with more reporting banks would be over-weighted
    in the threshold calculation.
    """
    unique_q = df[["rssd9999", "nfci"]].drop_duplicates(subset=["rssd9999"])
    threshold = unique_q["nfci"].quantile(threshold_pct)
    df["high_stress"] = (df["nfci"] >= threshold).astype(int)
    return df, threshold


def _fit_model_E(df, h):
    """
    Stress-dependent triple interaction: bank + quarter FE, testing
    whether the shock x L_it transmission-heterogeneity effect is
    amplified during high-financial-stress quarters (NFCI top tercile).
    Estimable terms under quarter FE (mp_shock and high_stress alone,
    and their product, are constant within a quarter and absorbed):
        L_it                        (main effect)
        L_it x high_stress          (does the linkage-outcome relationship
                                      itself differ by stress regime?)
        mp_shock x L_it             (baseline transmission-heterogeneity
                                      effect, i.e. during CALM quarters)
        mp_shock x L_it x high_stress  (the STRESS DIFFERENTIAL -- the
                                      parameter of interest: is transmission
                                      heterogeneity amplified in stress?)
    Total effect during high-stress quarters = coef(shock_x_L) +
    coef(shock_x_L_x_stress); its SE requires the joint covariance of
    the two coefficients (computed in main() via the fitted result's
    covariance matrix, not just added in quadrature independently).
    """
    cols = [f"y_h{h}", "L_it", "mp_shock", "high_stress"] + CONTROLS
    est = df[["rssd9001", "rssd9999"] + cols].dropna().copy()
    est["shock_x_L"] = est["mp_shock"] * est["L_it"]
    est["L_x_stress"] = est["L_it"] * est["high_stress"]
    est["shock_x_L_x_stress"] = est["mp_shock"] * est["L_it"] * est["high_stress"]
    n_before_singleton = len(est)
    est = _drop_singletons(est)
    n_singleton_dropped = n_before_singleton - len(est)
    n_banks = est["rssd9001"].nunique()
    est = est.set_index(["rssd9001", "rssd9999"])
    y = est[f"y_h{h}"]
    X = est[["L_it", "L_x_stress", INTERACT_VAR, "shock_x_L_x_stress"] + CONTROLS]
    try:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        n_clusters_bank = est.index.get_level_values(0).nunique()
        n_clusters_time = est.index.get_level_values(1).nunique()
        if pd.isna(res.std_errors.get("shock_x_L_x_stress", np.nan)):
            print(f"    NOTE h={h}: two-way clustered SE is NaN for shock_x_L_x_stress (Model E) "
                  f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time}) "
                  f"-- falling back to one-way (bank) clustering")
            res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            res._cov_note = (f"FALLBACK: one-way (bank) clustered SE used -- two-way "
                             f"clustered covariance was degenerate (NaN) at this horizon "
                             f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time})")
        else:
            res._cov_note = None
        return res, len(est), n_banks, n_singleton_dropped
    except Exception as e:
        print(f"    Model E FAILED at h={h}: {e}")
        return None, len(est), n_banks, n_singleton_dropped


def _build_linkage_terciles(df):
    """
    Split banks into Low/Medium/High linkage-TYPE groups based on
    terciles of L_it_bankmean (each bank's full-sample average L_it),
    computed on the DISTINCT bank-level distribution -- each bank
    counted once, not weighted by how many quarters it appears in the
    panel. Complements Model D's finding that the persistent bank
    characteristic (not the noisy quarterly flow) is what matters:
    here we test directly whether high-linkage banks are fundamentally
    different institutions, by comparing separately-estimated
    transmission coefficients across groups rather than an interaction
    term.
    """
    bank_level = df[["rssd9001", "L_it_bankmean"]].drop_duplicates(subset=["rssd9001"])
    q33, q67 = bank_level["L_it_bankmean"].quantile([1/3, 2/3])

    def _assign(x):
        if x <= q33:
            return "Low"
        elif x <= q67:
            return "Medium"
        return "High"

    bank_group_map = bank_level.set_index("rssd9001")["L_it_bankmean"].apply(_assign)
    df["linkage_group"] = df["rssd9001"].map(bank_group_map)
    return df, q33, q67


def _fit_model_F(df, h, group):
    """
    Bank FE only (no quarter FE -- mirrors Model B's identification,
    since we need mp_shock's own coefficient here, not just an
    interaction), estimated SEPARATELY within one linkage-type tercile
    group's subsample. beta_h here is the group-specific average
    transmission effect; comparing it across Low/Medium/High groups
    directly tests whether highly-interconnected banks transmit policy
    differently, complementing the interaction-based tests in Models
    A-E. Standard sorted-portfolio approach in the bank-lending-channel
    literature (Kashyap-Stein 1995 sort banks into liquid/illiquid
    terciles the same way).
    """
    sub = df[df["linkage_group"] == group]
    cols = [f"y_h{h}", "mp_shock", "nfci", "term_spread"] + CONTROLS
    est = sub[["rssd9001", "rssd9999"] + cols].dropna().copy()
    n_before_singleton = len(est)
    est = _drop_singletons(est, drop_time_singletons=False)
    n_singleton_dropped = n_before_singleton - len(est)
    n_banks = est["rssd9001"].nunique()
    est = est.set_index(["rssd9001", "rssd9999"])
    y = est[f"y_h{h}"]
    X = est[["mp_shock", "nfci", "term_spread"] + CONTROLS]
    try:
        mod = PanelOLS(y, X, entity_effects=True, time_effects=False, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        n_clusters_bank = est.index.get_level_values(0).nunique()
        n_clusters_time = est.index.get_level_values(1).nunique()
        if pd.isna(res.std_errors.get("mp_shock", np.nan)):
            print(f"    NOTE h={h} group={group}: two-way clustered SE is NaN for mp_shock "
                  f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time}) "
                  f"-- falling back to one-way (bank) clustering")
            res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=False)
            res._cov_note = (f"FALLBACK: one-way (bank) clustered SE used, group={group}, h={h} "
                             f"(n_bank_clusters={n_clusters_bank}, n_time_clusters={n_clusters_time})")
        else:
            res._cov_note = None
        return res, len(est), n_banks, n_singleton_dropped
    except Exception as e:
        print(f"    Model F ({group}) FAILED at h={h}: {e}")
        return None, len(est), n_banks, n_singleton_dropped


def _stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def main():
    print("\n" + "=" * 65)
    print("04_regression_p1.py  --  P1 monetary policy transmission (local projections)")
    print("=" * 65)

    log = [
        "P1 LOCAL PROJECTIONS AUDIT LOG", "=" * 65,
        f"Horizons: h = {HORIZONS}",
        "Model A (primary): bank + quarter FE, contemporaneous L_it, two-way clustered SE",
        "Model B (robustness): bank FE only + nfci/term_spread, contemporaneous L_it, two-way clustered SE",
        "Model C (robustness): bank + quarter FE, L_{i,t-1} (lagged linkage, avoids simultaneity)",
        "Model D (robustness): bank + quarter FE, time-averaged L_it (\"linkage type\", avoids flow-measure noise)",
        "Model E (extension): bank + quarter FE, stress-dependent triple interaction "
        "(shock x L_it x high_stress, NFCI top tercile)",
        "",
    ]

    # Load and build horizon outcomes #
    print("\n[1] Loading panel and constructing horizon outcomes")
    df = pd.read_parquet(config.F_PANEL_MACRO)
    df["rssd9999"] = pd.to_datetime(df["rssd9999"])
    df = _build_horizon_outcomes(df)
    df = _build_alt_linkage_vars(df)
    df, stress_threshold = _build_stress_indicator(df)
    n_stress_q = df.loc[df["high_stress"] == 1, "rssd9999"].nunique()
    n_total_q = df["rssd9999"].nunique()
    print(f"  High-stress threshold (NFCI top tercile): {stress_threshold:.4f}")
    print(f"  High-stress quarters: {n_stress_q} of {n_total_q}")
    log.append(f"High-stress threshold (NFCI >= top-tercile value {stress_threshold:.4f}): "
               f"{n_stress_q} of {n_total_q} quarters flagged")
    df, q33, q67 = _build_linkage_terciles(df)
    group_bank_counts = df.drop_duplicates(subset=["rssd9001"])["linkage_group"].value_counts()
    print(f"  Linkage-type terciles (L_it_bankmean): Low <= {q33:.4f} < Medium <= {q67:.4f} < High")
    print(f"  Bank counts by group: {group_bank_counts.to_dict()}")
    log.append(f"Linkage-type terciles: Low<={q33:.4f}, Medium<={q67:.4f}, High>{q67:.4f}; "
               f"bank counts: {group_bank_counts.to_dict()}")
    n_lag_valid = df["L_it_lag1"].notna().sum()
    print(f"  L_it_lag1: {n_lag_valid:,} valid obs (first quarter per bank has no lag)")
    log.append(f"L_it_lag1 valid obs: {n_lag_valid:,} of {len(df):,}")
    print(f"  Panel: {len(df):,} rows, {df['rssd9001'].nunique()} banks")
    for h in HORIZONS:
        n_valid = df[f"y_h{h}"].notna().sum()
        print(f"  h={h}: {n_valid:,} valid outcome obs")
        log.append(f"  h={h}: {n_valid:,} valid outcome observations (of {len(df):,})")

    # Estimate all five models at every horizon #
    print("\n[2] Estimating Models A-E across horizons")
    results_A, results_B, results_C, results_D, results_E = {}, {}, {}, {}, {}
    for h in HORIZONS:
        res_A, n_A, banks_A, dropped_A = _fit_model_A(df, h)
        res_B, n_B, banks_B = _fit_model_B(df, h)
        res_C, n_C, banks_C, dropped_C = _fit_model_C(df, h)
        res_D, n_D, banks_D, dropped_D = _fit_model_D(df, h)
        res_E, n_E, banks_E, dropped_E = _fit_model_E(df, h)
        results_A[h] = {"res": res_A, "n": n_A, "banks": banks_A}
        results_B[h] = {"res": res_B, "n": n_B, "banks": banks_B}
        results_C[h] = {"res": res_C, "n": n_C, "banks": banks_C}
        results_D[h] = {"res": res_D, "n": n_D, "banks": banks_D}
        results_E[h] = {"res": res_E, "n": n_E, "banks": banks_E}
        if dropped_A > 0:
            log.append(f"  h={h}: dropped {dropped_A} singleton obs before Model A (bank/quarter FE)")
        if dropped_C > 0:
            log.append(f"  h={h}: dropped {dropped_C} singleton obs before Model C")
        if dropped_D > 0:
            log.append(f"  h={h}: dropped {dropped_D} singleton obs before Model D")
        if dropped_E > 0:
            log.append(f"  h={h}: dropped {dropped_E} singleton obs before Model E")
        for label, res in [("Model A", res_A), ("Model B", res_B), ("Model C", res_C),
                           ("Model D", res_D), ("Model E", res_E)]:
            if res is not None and getattr(res, "_cov_note", None):
                log.append(f"  h={h} {label}: {res._cov_note}")
        line = f"  h={h}"
        for label, res, n in [("A", res_A, n_A), ("B", res_B, n_B), ("C", res_C, n_C), ("D", res_D, n_D)]:
            if res is not None:
                coef = res.params[INTERACT_VAR]
                pval = res.pvalues[INTERACT_VAR]
                line += f"  {label}: delta_h={coef:+.4f}{_stars(pval)} (N={n:,})"
            else:
                line += f"  {label}: failed"
        print(line)
        if res_E is not None:
            calm = res_E.params[INTERACT_VAR]
            calm_p = res_E.pvalues[INTERACT_VAR]
            stress_diff = res_E.params["shock_x_L_x_stress"]
            stress_diff_p = res_E.pvalues["shock_x_L_x_stress"]
            print(f"       E: calm={calm:+.4f}{_stars(calm_p)}  "
                  f"stress_differential={stress_diff:+.4f}{_stars(stress_diff_p)} (N={n_E:,})")
        else:
            print("       E: failed")

    # Build results table #
    print("\n[3] Building results table")
    rows = []
    for h in HORIZONS:
        for model_name, results in [("A", results_A), ("B", results_B), ("C", results_C), ("D", results_D)]:
            r = results[h]
            if r["res"] is None:
                rows.append({"model": model_name, "h": h, "coef": np.nan, "se": np.nan,
                            "pval": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                            "n": r["n"], "banks": r["banks"]})
                continue
            res = r["res"]
            coef  = res.params[INTERACT_VAR]
            se    = res.std_errors[INTERACT_VAR]
            pval  = res.pvalues[INTERACT_VAR]
            ci    = res.conf_int().loc[INTERACT_VAR]
            rows.append({
                "model": model_name, "h": h, "coef": coef, "se": se, "pval": pval,
                "ci_lo": ci["lower"], "ci_hi": ci["upper"],
                "n": r["n"], "banks": r["banks"], "stars": _stars(pval),
                "cov_note": getattr(res, "_cov_note", None),
            })
    df_csv = pd.DataFrame(rows)
    df_csv.to_csv(config.OUTPUT / "table_p1_local_projections.csv", index=False, float_format="%.6f")
    print(f"  CSV saved -> {config.OUTPUT / 'table_p1_local_projections.csv'}")

    # Formatted text table #
    print("\n[4] Writing formatted text table")
    txt_lines = [
        "TABLE: P1 Local Projections -- delta_h coefficient on (shock x linkage)",
        "Outcome: Delta_h y_it = log(CI_{i,t+h}) - log(CI_{i,t-1})",
        f"Sample: FFIEC 031 banks | Horizons h = 0..{max(HORIZONS)} quarters",
        "",
        f"{'h':<4}{'A: contemp L_it':>18}{'B: contemp,no-t.FE':>20}"
        f"{'C: lagged L':>16}{'D: bank-mean L':>16}",
        "-" * 74,
    ]
    for h in HORIZONS:
        line = f"{h:<4}"
        for results in [results_A, results_B, results_C, results_D]:
            r = results[h]
            if r["res"] is not None:
                c = r["res"].params[INTERACT_VAR]
                p = r["res"].pvalues[INTERACT_VAR]
                dagger = "\u2020" if getattr(r["res"], "_cov_note", None) else ""
                s = f"{c:+.4f}{_stars(p)}{dagger}"
            else:
                s = "failed"
            line += f"{s:>18}" if results is results_A else f"{s:>16}"
        txt_lines.append(line)

    txt_lines += ["-" * 74, "", "N by horizon (Model A / B / C / D):"]
    for h in HORIZONS:
        ns = [str(results[h]["n"]) for results in [results_A, results_B, results_C, results_D]]
        txt_lines.append(f"  h={h}: " + " / ".join(ns))

    txt_lines += [
        "", "-" * 74, "",
        "\u2020 Two-way (bank, quarter) clustered SE was degenerate (NaN) at this horizon;",
        "  one-way (bank) clustered SE reported instead. See p1_regression_log.txt for",
        "  cluster counts. Coefficient (point estimate) is unaffected either way.",
        "",
        "Notes: Coefficient shown is delta_h, on (mp_shock_t x linkage variable). Two-way",
        "clustered SEs (bank, quarter) unless noted. *** p<0.01  ** p<0.05  * p<0.10.",
        "Model A (PRIMARY): bank + quarter FE, contemporaneous L_it. mp_shock_t is",
        "  absorbed by quarter FE; delta_h is the only estimable parameter of interest.",
        "Model B (robustness): bank FE only, + nfci/term_spread, contemporaneous L_it.",
        "  Literal proposal-spec check; both beta_h and delta_h estimable, weaker ID.",
        "Model C (robustness): bank + quarter FE, L_{i,t-1} instead of contemporaneous",
        "  L_it -- avoids simultaneity between the shock and the linkage measure within",
        "  the same quarter (Kashyap-Stein 1995, Kishan-Opiela 2000).",
        "Model D (robustness): bank + quarter FE, time-averaged L_it (\"linkage type\")",
        "  instead of the noisy quarterly flow measure -- avoids attenuation from",
        "  quarter-to-quarter measurement noise (L_it has a large mass at exactly zero).",
        "mp_shock is a quarterly SUM of the Jarocinski-Karadi monthly MP_pm shock --",
        "  identifies the effect of the quarterly-cumulated shock, not a single FOMC",
        "  announcement (frequency-mismatch limitation, see 03_macro_merge.py).",
        "Because the outcome is cumulative from t-1, delta_h at h=8 IS the cumulative",
        "  eight-quarter effect (no further summation needed).",
    ]
    txt = "\n".join(txt_lines)
    (config.OUTPUT / "table_p1_local_projections.txt").write_text(txt, encoding="utf-8")
    print(f"  TXT saved -> {config.OUTPUT / 'table_p1_local_projections.txt'}")
    print("\n" + txt)

    # Peak effect / time-to-peak / cumulative effect summary #
    print("\n[5] Peak effect and cumulative-effect summary (all models)")
    for label, results in [("A (primary)", results_A), ("B", results_B),
                           ("C (lagged L)", results_C), ("D (bank-mean L)", results_D)]:
        coefs = {h: results[h]["res"].params[INTERACT_VAR]
                 for h in HORIZONS if results[h]["res"] is not None}
        if not coefs:
            continue
        peak_h = max(coefs, key=lambda h: abs(coefs[h]))
        peak_val = coefs[peak_h]
        cum_8 = coefs.get(8, np.nan)
        summary_lines = [
            "", f"PEAK EFFECT / TIME-TO-PEAK / CUMULATIVE EFFECT (Model {label})",
            f"  Peak |delta_h|: h={peak_h}, delta_{peak_h}={peak_val:+.4f}",
            f"  Cumulative 8-quarter effect: delta_8={cum_8:+.4f}" if not np.isnan(cum_8)
                else "  Cumulative 8-quarter effect: NOT AVAILABLE (insufficient N at h=8)",
        ]
        for line in summary_lines:
            print(line)
            log.append(line)

    # Model E: stress-dependent results (dedicated table) #
    print("\n[5b] Model E: stress-dependent triple interaction results")
    e_rows = []
    e_lines = [
        "TABLE: P1 Extension -- Stress-Dependent Triple Interaction (Model E)",
        f"High-stress = NFCI in top tercile (threshold={stress_threshold:.4f})",
        "Bank + quarter FE throughout. mp_shock, high_stress, and their product",
        "are absorbed by quarter FE; the four estimable terms are shown below.", "",
        f"{'h':<4}{'calm (shock x L)':>18}{'stress diff.':>16}{'total (stress)':>16}{'total SE':>10}", "-" * 68,
    ]
    for h in HORIZONS:
        res = results_E[h]["res"]
        if res is None:
            e_lines.append(f"{h:<4}{'failed':>18}")
            continue
        calm = res.params[INTERACT_VAR]
        calm_se = res.std_errors[INTERACT_VAR]
        calm_p = res.pvalues[INTERACT_VAR]
        stress_diff = res.params["shock_x_L_x_stress"]
        stress_diff_p = res.pvalues["shock_x_L_x_stress"]
        total = calm + stress_diff
        # Joint SE via the fitted covariance matrix (NOT independent addition-in-quadrature,
        # since the two coefficients are correlated in finite samples)
        try:
            cov_matrix = res.cov
            var_total = (cov_matrix.loc[INTERACT_VAR, INTERACT_VAR]
                        + cov_matrix.loc["shock_x_L_x_stress", "shock_x_L_x_stress"]
                        + 2 * cov_matrix.loc[INTERACT_VAR, "shock_x_L_x_stress"])
            total_se = np.sqrt(var_total) if var_total > 0 else np.nan
        except Exception:
            total_se = np.nan
        if not np.isnan(total_se) and total_se > 0:
            total_tstat = total / total_se
            total_pval = 2 * (1 - stats.norm.cdf(np.abs(total_tstat)))
        else:
            total_pval = np.nan
        e_lines.append(
            f"{h:<4}{f'{calm:+.4f}{_stars(calm_p)}':>18}{f'{stress_diff:+.4f}{_stars(stress_diff_p)}':>16}"
            f"{f'{total:+.4f}{_stars(total_pval)}':>16}{total_se:>10.4f}"
        )
        e_rows.append({"h": h, "calm": calm, "calm_se": calm_se, "calm_pval": calm_p,
                       "stress_differential": stress_diff, "stress_differential_pval": stress_diff_p,
                       "total_high_stress": total, "total_high_stress_se": total_se,
                       "total_high_stress_pval": total_pval, "n": results_E[h]["n"]})
    e_lines += [
        "-" * 68, "",
        "Notes: 'calm' = coefficient on (mp_shock x L_it), i.e. the transmission-",
        "  heterogeneity effect during LOW-stress quarters (the baseline).",
        "'stress diff.' = coefficient on (mp_shock x L_it x high_stress), i.e. how much",
        "  MORE (or less) the effect is during HIGH-stress quarters.",
        "'total (stress)' = calm + stress diff. = the total effect during high-stress",
        "  quarters; its SE uses the joint covariance of the two coefficients, not",
        "  independent addition-in-quadrature.",
        "Motivated by the dissertation plan's own Brunnermeier-Pedersen funding-stress",
        "  channel discussion (Proposition 2) -- tests whether P1's null in Models A-C",
        "  masks a real effect concentrated in stress periods, averaged against a",
        "  near-zero effect in calm periods.",
    ]
    e_text = "\n".join(e_lines)
    (config.OUTPUT / "table_p1_stress_dependent.txt").write_text(e_text, encoding="utf-8")
    pd.DataFrame(e_rows).to_csv(config.OUTPUT / "table_p1_stress_dependent.csv", index=False, float_format="%.6f")
    print(f"  Saved -> {config.OUTPUT / 'table_p1_stress_dependent.txt'}")
    print("\n" + e_text)

    # Impulse response figure #
    print("\n[6] Generating impulse response figure")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    def _plot_panel(ax, specs, title):
        for results, color, label in specs:
            hs, coefs, ci_lo, ci_hi = [], [], [], []
            for h in HORIZONS:
                r = results[h]["res"]
                if r is None:
                    continue
                hs.append(h)
                coefs.append(r.params[INTERACT_VAR])
                ci = r.conf_int().loc[INTERACT_VAR]
                ci_lo.append(ci["lower"])
                ci_hi.append(ci["upper"])
            if hs:
                ax.plot(hs, coefs, color=color, lw=2, marker="o", markersize=5, label=label)
                ax.fill_between(hs, ci_lo, ci_hi, color=color, alpha=0.15)
        ax.axhline(0, color=GRAY, lw=1, ls="--")
        ax.set_xlabel("Horizon h (quarters)")
        ax.set_ylabel("delta_h: coefficient on (shock x linkage)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_xticks(HORIZONS)

    _plot_panel(axes[0],
        [(results_A, NAVY, "A: bank+quarter FE (primary)"),
         (results_B, BLUE, "B: bank FE only (robustness)")],
        "Identification comparison\n(contemporaneous L_it)")
    _plot_panel(axes[1],
        [(results_A, NAVY, "A: contemporaneous L_it"),
         (results_C, RED, "C: lagged L_{i,t-1}"),
         (results_D, "#006633", "D: bank-mean L (\"type\")")],
        "Linkage-variable comparison\n(bank+quarter FE throughout)")

    fig.suptitle("P1: Does linkage intensity alter monetary policy transmission?", y=1.03)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_p1_impulse_response.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {config.OUTPUT / 'fig_p1_impulse_response.png'}")

    # ── [7] Model E figure: calm vs stress-differential vs total ────────
    print("\n[7] Generating stress-dependent (Model E) figure")
    hs_e, calm_vals, stress_diff_vals, total_vals, total_ci_lo, total_ci_hi = [], [], [], [], [], []
    for h in HORIZONS:
        res = results_E[h]["res"]
        if res is None:
            continue
        hs_e.append(h)
        calm_vals.append(res.params[INTERACT_VAR])
        stress_diff_vals.append(res.params["shock_x_L_x_stress"])
        total = res.params[INTERACT_VAR] + res.params["shock_x_L_x_stress"]
        total_vals.append(total)
        try:
            cov_matrix = res.cov
            var_total = (cov_matrix.loc[INTERACT_VAR, INTERACT_VAR]
                        + cov_matrix.loc["shock_x_L_x_stress", "shock_x_L_x_stress"]
                        + 2 * cov_matrix.loc[INTERACT_VAR, "shock_x_L_x_stress"])
            se_total = np.sqrt(var_total) if var_total > 0 else np.nan
        except Exception:
            se_total = np.nan
        total_ci_lo.append(total - 1.96 * se_total)
        total_ci_hi.append(total + 1.96 * se_total)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(hs_e, calm_vals, color=BLUE, lw=2, marker="o", markersize=5, label="Calm quarters (baseline)")
    ax.plot(hs_e, total_vals, color=RED, lw=2, marker="s", markersize=5, label="High-stress quarters (total effect)")
    ax.fill_between(hs_e, total_ci_lo, total_ci_hi, color=RED, alpha=0.15)
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel("Coefficient on (mp_shock x L_it)")
    ax.set_title("Does transmission heterogeneity depend on financial stress?\n"
                 "(NFCI top-tercile quarters vs. calm quarters)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(HORIZONS)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_p1_stress_dependent.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {config.OUTPUT / 'fig_p1_stress_dependent.png'}")

    # ══════════════════════════════════════════════════════════════════
    # [8] Model F: linkage-type tercile group comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n[8] Model F: linkage-type tercile group comparison")
    groups = ["Low", "Medium", "High"]
    results_F = {g: {} for g in groups}
    for h in HORIZONS:
        for g in groups:
            res, n_g, banks_g, dropped_g = _fit_model_F(df, h, g)
            results_F[g][h] = {"res": res, "n": n_g, "banks": banks_g}
            if dropped_g > 0:
                log.append(f"  h={h} group={g}: dropped {dropped_g} singleton obs before Model F")
            if res is not None and getattr(res, "_cov_note", None):
                log.append(f"  h={h} group={g}: {res._cov_note}")
        line = f"  h={h}"
        for g in groups:
            r = results_F[g][h]["res"]
            if r is not None:
                line += f"  {g}: beta_h={r.params['mp_shock']:+.4f}{_stars(r.pvalues['mp_shock'])} (N={results_F[g][h]['n']:,})"
            else:
                line += f"  {g}: failed"
        print(line)

    f_rows = []
    f_lines = [
        "TABLE: P1 Extension -- Linkage-Type Tercile Group Comparison",
        f"Groups by L_it_bankmean: Low<={q33:.4f}, Medium<={q67:.4f}, High>{q67:.4f}",
        f"Bank counts: {group_bank_counts.to_dict()}",
        "Bank FE only (no quarter FE -- need mp_shock's own coefficient, not just an",
        "interaction), estimated SEPARATELY within each group's subsample.", "",
        f"{'h':<4}{'Low':>16}{'Medium':>16}{'High':>16}{'High-Low diff':>16}{'p-value':>10}", "-" * 78,
    ]
    for h in HORIZONS:
        row = f"{h:<4}"
        coefs = {}
        for g in groups:
            r = results_F[g][h]["res"]
            if r is not None:
                c = r.params["mp_shock"]
                p = r.pvalues["mp_shock"]
                coefs[g] = (c, r.std_errors["mp_shock"])
                row += f"{f'{c:+.4f}{_stars(p)}':>16}"
            else:
                row += f"{'failed':>16}"
        if "High" in coefs and "Low" in coefs:
            diff = coefs["High"][0] - coefs["Low"][0]
            # Low and High are non-overlapping, independent bank subsamples --
            # variances simply add (no covariance term needed, unlike Model E's
            # within-sample joint interaction terms)
            diff_se = np.sqrt(coefs["High"][1] ** 2 + coefs["Low"][1] ** 2)
            diff_t = diff / diff_se if diff_se > 0 else np.nan
            diff_p = 2 * (1 - stats.norm.cdf(np.abs(diff_t))) if not np.isnan(diff_t) else np.nan
            row += f"{f'{diff:+.4f}{_stars(diff_p)}':>16}{diff_p:>10.4f}"
            f_rows.append({"h": h, "low": coefs.get("Low", (np.nan,))[0],
                          "medium": coefs.get("Medium", (np.nan,))[0],
                          "high": coefs.get("High", (np.nan,))[0],
                          "high_minus_low": diff, "high_minus_low_se": diff_se, "high_minus_low_pval": diff_p})
        else:
            row += f"{'n/a':>16}{'n/a':>10}"
        f_lines.append(row)
    f_lines += [
        "-" * 78, "",
        "Notes: beta_h = coefficient on mp_shock, estimated separately within each",
        "  linkage-type tercile's subsample (bank FE only, two-way clustered SE).",
        "'High-Low diff' = High-group beta_h minus Low-group beta_h; independent",
        "  (non-overlapping) subsamples, so its SE is simply sqrt(se_High^2 + se_Low^2).",
        "*** p<0.01  ** p<0.05  * p<0.10 (two-sided, standard normal).",
        "Complements Model D (bank-mean L_it as an interaction): here the comparison is",
        "  between separately-estimated coefficients across sorted groups, the standard",
        "  approach in the bank-lending-channel literature (Kashyap-Stein 1995).",
    ]
    f_text = "\n".join(f_lines)
    (config.OUTPUT / "table_p1_linkage_groups.txt").write_text(f_text, encoding="utf-8")
    pd.DataFrame(f_rows).to_csv(config.OUTPUT / "table_p1_linkage_groups.csv", index=False, float_format="%.6f")
    print(f"  Saved -> {config.OUTPUT / 'table_p1_linkage_groups.txt'}")
    print("\n" + f_text)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    group_colors = {"Low": GREEN, "Medium": BLUE, "High": RED}
    for g in groups:
        hs_g, coefs_g, ci_lo_g, ci_hi_g = [], [], [], []
        for h in HORIZONS:
            r = results_F[g][h]["res"]
            if r is None:
                continue
            hs_g.append(h)
            coefs_g.append(r.params["mp_shock"])
            ci = r.conf_int().loc["mp_shock"]
            ci_lo_g.append(ci["lower"])
            ci_hi_g.append(ci["upper"])
        if hs_g:
            ax.plot(hs_g, coefs_g, color=group_colors[g], lw=2, marker="o", markersize=5,
                    label=f"{g} linkage-type")
            ax.fill_between(hs_g, ci_lo_g, ci_hi_g, color=group_colors[g], alpha=0.12)
    ax.axhline(0, color=GRAY, lw=1, ls="--")
    ax.set_xlabel("Horizon h (quarters)")
    ax.set_ylabel("beta_h: coefficient on mp_shock")
    ax.set_title("Are highly-interconnected banks fundamentally different?\n"
                 "Transmission effect estimated separately by linkage-type tercile")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(HORIZONS)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "fig_p1_linkage_groups.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {config.OUTPUT / 'fig_p1_linkage_groups.png'}")

    # Save log #
    log_text = "\n".join(log)
    (config.OUTPUT / "p1_regression_log.txt").write_text(log_text, encoding="utf-8")
    print(f"\n  Log -> {config.OUTPUT / 'p1_regression_log.txt'}")
    print("\n  Done. Next: python 05_regression_p2.py")


if __name__ == "__main__":
    main()