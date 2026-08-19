<h2>Notes:</h2>
1) Used .gitignore to hide proprietary raw data

<h2>Libraries/Requirements:</h2>
The pipeline is implemented entirely in Python. Data handling and panel construction rely on pandas and numpy, with pyarrow required as the Parquet engine for the intermediate panel files; all figures are produced with matplotlib. Estimation draws on statsmodels for quantile regression, linearmodels for panel fixed-effects models with multi-way clustering, and scipy for bootstrap inference and hypothesis testing. Supplementary data collection uses requests for the SEC EDGAR API and rapidfuzz for DealScan lender-name matching, and openpyxl is required by pandas to read the Capital IQ Excel export.

<ul>
  <li>pandas</li>
  <li>numpy</li>
  <li>pyarrow (Parquet I/O)</li>
  <li>matplotlib</li>
  <li>scipy</li>
  <li>statsmodels</li>
  <li>linearmodels</li>
  <li>requests (07a_sec_bdc_json.py only)</li>
  <li>rapidfuzz (08_dealscan_lender_match.py only)</li>
  <li>openpyxl (required by pandas to read .xlsx files)</li>
  <li>seaborn</li>
</ul>

<h2>Data & Pipeline Notes:</h2>

Supplementary notes on data provenance, known issues, and decisions made during pipeline construction. Organised by script.

<h3>00_config.py</h3>

Central config. Every other script imports from here (via importlib, since filenames start with digits) rather than hardcoding paths or thresholds, so a change only has to happen once.

**Pipeline history.** Originally built around a 4-file WRDS pull (rcfd_old.csv + 3 patch files) because I am not bright and was too tired to repull before knowing what I am getting into. That legacy file is gone now, and the consolidated 2-file pull (rcfd1.csv + rcfd2.csv) covers every core L_it component for FFIEC 031 filers from 2004Q4 onward, which is the whole window needed. Fields are split across two files only because WRDS's query builder caps the column count per request.

**RCFD1520 / RCONJ454.** RCFD1520 (drawn loans to non-depository FIs is the item I actually wanted at the start of this) is not available under my WRDS subscription tier (Bank Regulatory Premium). I tried to pull it directly from FRB Chicago, but turns out they had a split in conventions to RCFDJ454 and more granular format, which made me reconsider. I tried RCFDJ454 looked like a possible substitute but is zero-populated for almost the entire sample (only starts 2024Q4, and only for the largest banks), but it was not not usable. RCONJ454 turned out to be the domestic-offices counterpart of the same underlying MDRM concept, with continuous coverage from 2010Q1. Confirmed by comparing FFIEC's own bulk Call Report files for 2010Q1 and 2024Q4 side by side: RCFD1520 wasn't in either one (retired before 2010), so this isn't WRDS protected behind a paywall. For this US$ 300 mln+ sample, none of which have foreign offices, RCON and RCFD bases are numerically identical, so using RCONJ454 isn't a compromise. Fortunate outcome for me.

**Sample window.** Originally ran through 2025Q4. Trimmed back to 2024Q3 after finding FFIEC retired RCFDJ458 ("unused commitments to FIs") starting the 2024Q4 filing cycle and checked directly against the FFIEC bulk Call Report schedule for Dec 2024, the column simply isn't in the RC-L schedule anymore. RCFDJ457/459 are still there but got repurposed to mean drawn loan balances rather than unused commitments, so they can't substitute either. Write it off as an item retirement.

**rcfd3818 (unused commitments — other, aggregate).** Deliberately not included in L_it. For a handful of FFIEC 031 trust-company filers it comes out larger than total assets, which would poison L_it with commitments unrelated to NBFI exposure. FFIEC retired this item in 2010 anyway, in favour of the rcfdj457/458/459 breakdown, so excluding it lines up with how the reporting form itself evolved. Kept in the cleaned panel for reference, just not used in the index. 

**2005Q3 dip in bank coverage.** Documented with numbers in 01_data_cleaning.py, confirmed against the final panel rather than an earlier pipeline stage. Coincides with the CDR electronic filing system going live and Hurricane Katrina in the same reporting period and found validation of this.

**DealScan crosswalk.** lender_parent_id to RSSD mapping built by hand-reviewing fuzzy name matches (08_dealscan_lender_match.py). A handful of entries dropped on purpose: a few structurally ambiguous parent entities (e.g. one DealScan parent covering two legally separate subsidiary banks), and two merger-timing cases where the current entity name doesn't reliably map to one RSSD across the whole sample window. I reviewed all 150+ entities manually, confirming them each through google search and company filings if I could. A very exhausting task.

**BDC data.** Primary source Capital IQ, pulled across the whole BDC industry classification (67 tickers resolved, quarterly, 2010–2025). Cross-checked against SEC XBRL company-facts data (38 tickers, patchier coverage), only using the LineOfCreditFacilityMaximumBorrowingCapacity tag, since it's the one field that unambiguously means "bank credit facility capacity". Other tags (DebtInstrumentCarryingAmount etc.) are too generic to trust without more validation work than this cross-check is worth.

<h3>01_data_cleaning.py</h3>

Loads the two WRDS Call Report pulls, dedupes, merges, and applies the sample-definition filters. Produces the clean bank-quarter panel that 02_linkage_index.py builds L_it on top of. 

**Duplicate (bank, quarter) keys with conflicting values.** A handful of (rssd9001, rssd9999) keys appear more than once with different reported values within the same file — most likely original vs. amended filings. WRDS doesn't expose a submission-date field in this pull, so there's no certain way to pick the "correct" version. These keys are dropped from both files before merging, rather than kept and arbitrarily resolved, which keeps the merge a clean 1:1 join at the cost of ~1% of bank-quarters.

**2005Q3 dip in bank coverage.** The panel shows a visible drop in bank count for 2005Q3 relative to the surrounding quarters. Q3 2005 was the first quarter Call Reports were processed through the newly-implemented CDR electronic filing system, and Hurricane Katrina struck during the same reporting quarter and the agencies formally acknowledged resulting filing delays (FDIC/OCC/Fed joint filing FIL-94-2005, Sept 21 2005). Left as gap.

**Items not zero-filled.** rcfd3818 (retired 2010) and the NDFI-breakdown items (rcfdj454, pv-series, only required from 2024Q4) follow different reporting rules than the rest of Schedule RC-L, so a missing value there doesn't mean "zero" the way it does for the other RC-L items as these are left as NaN rather than zero-filled.

Full audit trail (row/bank counts at each step, coverage percentages) is written to `outputs/cleaning_log.txt` at runtime.

<h3>02_linkage_index.py</h3>

Constructs the bank-NBFI linkage index (L_it) and its component variants from the cleaned panel, runs validation checks, and builds a concentration diagnostic (suggested by my supervisor).

**credit_growth_4q.** Built as a calendar-based self-merge  rather than a simple shift(4) within the panel and is robust to panel gaps.

**Concentration diagnostic correlation cutoff.** The |corr| > 0.3 threshold used to flag "L_it itself is concentrated in the largest banks" is a standard rule-of-thumb for "not a strong correlation" and not derived from anything, just a reasonable line to draw for the write-up. Referenced as a footnote in `outputs/table_concentration.txt`.

<h3>03_macro_merge.py</h3>

Loads the manually-downloaded FRED CSVs (NFCI, NFCICREDIT, DGS10, FEDFUNDS) and the Jarocinski-Karadi monthly shock series, aggregates both to quarterly frequency, and merges into the clean panel.

**Frequency mismatch.** NFCI/NFCICREDIT (weekly), DGS10 (daily), and FEDFUNDS (monthly) are all aggregated to quarterly means to match the bank-quarter panel. This discards within-quarter timing, which is standard practice in the bank-lending literature, but the same limitation applies to the JK shock series itself, which is the actual regressor of interest in Proposition 1. The local projections in 04_regression_p1.py therefore identify the effect of the quarterly-cumulated shock. Stated explicitly in the methodology section of the main paper.

<h3>04_regression_p1.py</h3>

Estimates the Proposition 1 local projections (Jorda, 2005): does bank-NBFI linkage intensity (L_it) alter the strength of monetary policy transmission to bank credit growth, at horizons h = 0.8 quarters.
Outcome: Delta_h y_{i,t} = log(CI_{i,t+h}) − log(CI_{i,t-1}), built via calendar-based merges for the same panel-gap-robustness reason as `credit_growth_4q` in 02_linkage_index.py.

**Horizon-outcome merge bug.** The correct pattern shifts the <i>target</i> date on the main row and merges against an unshifted copy of: bank, date, ci_level. An earlier version shifted the source copy instead, which silently computes values from h quarters in the past rather than the future. Caught because y_h1 had exactly zero variance in testing (h=0 is shift-symmetric, so it happened to look fine).

**Models.**
- **A (primary):** bank + quarter FE, contemporaneous L_it. mp_shock_t is absorbed by the quarter FE which is constant across banks within a quarter, so only the interaction delta_h, is estimable here (not the avg transmission effect of beta_h). This is the tighter identification as quarter FE absorb any common shock that quarter, so delta_h isn't contaminated by omitted aggregate confounds.
- **B (robustness):** bank FE only + nfci/term_spread as aggregate controls (no quarter FE to absorb them). Matches the dissertation plan's literal spec (both beta_h and delta_h estimable) at the cost of weaker identification, since aggregate confounds not captured by the two controls contaminate both coefficients.
- **C (robustness):** bank + quarter FE, interacts the shock with L_{i,t-1} instead of contemporaneous L_it, to avoid simultaneity (the shock itself could move L_it within the same quarter). Standard in the bank-lending-channel literature (Kashyap-Stein 1995, Kishan-Opiela 2000).
- **D (robustness):** bank + quarter FE, interacts the shock with each bank's full-sample average L_it (a time-invariant "linkage type") instead of the noisy quarterly flow measure. L_it has a large mass at exactly zero and moves in/out of securities lending quarter to quarter; the contemporaneous value can mechanically attenuate the interaction toward zero even if a real effect exists.
- **E (extension):** bank + quarter FE, adds a stress-dependent triple interaction (shock x L_it x high_stress, high_stress = NFCI top historical tercile). Motivated by the dissertation plan's own Brunnermeier-Pedersen funding-stress discussion (central to Proposition 2) and by Models A-C's unconditional null: if the effect is real but concentrated in stress periods, an unconditional average against near-zero calm-period effects looks null even though a genuine state-dependent effect exists.
- **F (extension):** same identification as Model B (bank FE only, need mp_shock's own coefficient), but estimated separately within Low/Medium/High linkage-type tercile subsamples (terciles of L_it_bankmean) rather than as an interaction. Standard sorted-portfolio approach (Kashyap-Stein 1995 sort banks into liquid/illiquid terciles the same way); complements Model D by testing directly whether high-linkage banks are fundamentally different institutions.

**Two-way clustered SE fallback.** Two-way (bank, quarter) cluster-robust covariance (Cameron-Gelbach-Miller) can be numerically degenerate without returning NaN as it can instead yield an artificially deflated SE that looks like a real, spuriously significant result. Checked directly if the two-way SE is under 20% of the one-way (bank-clustered) SE, that's treated as a red flag for degenerate covariance rather than genuine extra precision, and the model falls back to one-way clustering. Flagged in the tables with a dagger symbol (†) and logged with cluster counts.

**Model E total-effect SE.** The total effect during high-stress quarters (calm + stress differential) uses the joint covariance of the two coefficients from the fitted covariance matrix, since the two terms are correlated in finite samples. Model F's High-minus-Low difference, by contrast, is between non-overlapping bank subsamples, so its SE is a simple sqrt(se_High² + se_Low²).

<h3>05_regression_p2.py</h3>

Estimates the Proposition 2 Growth-at-Risk quantile regression: does bank-NBFI linkage intensity (L_it) predict downside tail risk in four-quarter-ahead C&I credit growth, incremental to conventional prudential indicators? Tests three hypotheses from the dissertation plan (fragility, redundancy, asymmetry) via Model A (no L_it) vs Model B (+ L_it) across five quantiles.

**Inference.** Pairs cluster bootstrap (B=500, clustered on bank), keeping each bank's full time series intact within a replication and corrects for within-bank autocorrelation without parametric assumptions on the error structure. SEs are the empirical SD of the bootstrap distribution; 95% CIs are percentile-based; p-values are two-sided against the standard normal.

<h3>06_robustness.py</h3>

Four pre-specified robustness checks (see Methodology for full rationale): P2 with lagged linkage (L_{i,t-1}); P2 re-run separately by L_it component (L_middle, L_guarantees, L_securities, L_lines) to identify which piece drives the tail-risk result; P2 excluding, then restricted to, the top 10 banks by mean assets; P1 Model A excluding ZLB/COVID quarters (fedfunds ≤ 0.25%). Reuses the tested fitting logic from `04_regression_p1.py` and `05_regression_p2.py` as imported modules rather than reimplementing it.

<h3>07b_bdc_supplement.py</h3>

Descriptive BDC triangulation exhibit which isn't a merged regression, since bank-BDC pair-level linking isn't attempted. Tests whether aggregate BDC-sector leverage / bank-facility usage moves in a way consistent with the bank-side L_it story using independent data.

**Data sources.** Primary: Capital IQ BDC industry pull, 67 tickers, quarterly, 2010Q1–2024Q3 (matches the main panel window — see 00_config.py on the 2024Q4 RC-L retirement). Cross-check: SEC XBRL company-facts, 38 tickers, sparser coverage; only LineOfCreditFacilityMaximumBorrowingCapacity is used as an independent check on Capital IQ's revolving_credit + undrawn_credit, and DebtInstrumentCarryingAmount as a secondary check on total_debt (confirmed clean: one consolidated value per company-period, no fragmentation across instruments).

**Zero-as-missing.** Zero values in the Capital IQ pull are treated as missing rather than genuine zero, since a real operating public BDC doesn't report literally $0 total assets. Confirmed against individual tickers: some (e.g. ARCC) have no such gaps since 2010; others (e.g. BXSL, OBDC) show a clean zero-prefix ending exactly at their IPO quarter; but roughly 28 of 67 tickers show scattered zeros not explained by IPO timing alone, likely fiscal-year misalignment or merger-related ticker transitions. I treat zero as missing handles both cases for consistency.

**Triangulation correlation.** Reported as both the raw-levels correlation and the quarter-on-quarter change correlation between BDC-sector aggregate bank debt and bank-side mean L_lines. The levels correlation is reported for completeness but has a spurious-correlation risk, since both series trend upward over the sample from general private-credit-sector growth; the QoQ-change correlation, which netting out any shared trend, is the more defensible number to cite I believe.

<h3>08_dealscan_lender_match.py</h3>

Matches DealScan's Lender_Parent_Name/Lender_Parent_Id entities to RSSD bank IDs in the Call Report panel via fuzzy name matching, producing a candidate-match file for manual review. Since bank name matching needed a human confirmation, since fuzzy is very imperfect producing both false positives and negatives. Essentially I manually reviewed, and to make sure I used Gemini (due to it's data richness) in Google Sheets and used a gemini connected prompt =(AI, "...", cell reference(s)) to filter it down further before manually reviewing it (95% of them were correct).

**Known complication — retroactive parent names.** DealScan's Lender_Parent_Name reflects the *current* corporate structure applied retroactively to historical loans (e.g. "Truist Financial" appears as the parent for SunTrust Bank loans from 2010–2012, years before Truist existed as the Dec 2019 merger of BB&T and SunTrust). This script only produces a name match; a separate, later step builds a time-aware crosswalk mapping each DealScan parent to the RSSD ID that actually existed at each loan's date.

<h3>09_dealscan_supplement.py</h3>

Descriptive triangulation using loan-level DealScan data for the subset of the panel confidently matched to DealScan lenders (68 confirmed lender_parent_id to RSSD matches, covering 56 of the panel banks (see 08_dealscan_lender_match.py)). Scoped down from a full Khwaja-Mian (2008) firm-time fixed-effects design, which would require resolving merger-timing ambiguity for every confirmed match (not just the ones caught during review) and enough multi-lender same-quarter borrowers to identify firm-time FE, neither of which is here. What this script does is it compares loan-level pricing and volume for the confirmed-match banks against their Call-Report-based L_it and credit growth as corroborating descriptive evidence.

**Lender_Share correction.** Lender_Share is a percent of the facility, so a bank's actual dollar exposure is facility_amount × (Lender_Share / 100). Using the raw facility amount would credit every co-lender in a syndicated deal with the full facility size, double/triple-counting the same loan across multiple banks' aggregates.

**Date field.** Uses Tranche_Active_Date throughout, not Deal_Active_Date. Many rows show exactly 1/1/YYYY for deals with no precisely known date, confirmed via sample inspection during the initial pull.

**Role split.** Lead/Arranger-type roles (Admin agent, Arranger, Mandated Lead arranger, Lead arranger, Syndication agent, Bookrunner, Documentation, Managing agent, Co-agent) carry agency over loan terms; Participant roles bought into a deal arranged by someone else. Correlations are reported separately by role group as well as pooled, since the two roles reflect a different relationship to loan pricing.

<h3>10_linkage_orthogonality.py</h3>

Diagnostic (not a hypothesis test): how much of L_it is just a repackaged version of standard, already-observable bank characteristics (size, capital ratio, loan growth, asset growth), and how much is genuinely orthogonal to them? credit_growth_4q is excluded from the characteristics set since it's the P1/P2 outcome variable. Reports both panel-level (pooled bank-quarter) and bank-level (cross-sectional, full-sample bank means) correlations which I found can differ, since panel-level asks whether linkage moves with these characteristics quarter to quarter, while bank-level asks whether persistently high-linkage banks are a particular type of bank. Also reports the R^2 of a plain bank-level OLS of mean L_it on all four characteristics jointly: 1 − R^2 is the sharper answer to "how much is orthogonal," since a handful of individually-weak pairwise correlations can't rule out L_it being a linear combination of several observables together.

This is a plain OLS/correlation diagnostic, with no clustered SEs or bootstrap, unlike the P1/P2 models (04/05/06). A formal test of any individual relationship should use the same clustered-bootstrap approach as 05/06.

**has_sec_lending exclusion.** has_sec_lending is deliberately excluded from the characteristics set used for the headline correlation table and R^2. It is (rcfd3433 > 0), a binarized version of one of L_it's own four numerator components (please refer to 02_linkage_index.py). Folding it in would let a piece of L_it's own construction explain L_it, inflating "orthogonality" through circularity. It's reported separately as a construction check, since a strong correlation there is expected by construction and confirms internal consistency.