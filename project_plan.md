Final Executable Plan: Validation and Feature Prioritisation

The plan explicitly proceeds with Volume Bars (VB) due to their empirical superiority in volatility stability and normality on the BTC/POWR/ETC/SOL dataset, which contradicted the theoretical preference for Dollar Bars (DB).

--------------------------------------------------------------------------------

Phase 0: Data Structuring and Bar Selection (Completed)
Step
Action
Status
Key Chapter(s)
Key Finding/Rationale
0.1–0.6 Bar Comparison
Conduct statistical comparisons (Stability, Serial Correlation, Normality) across Tick, Volume, and Dollar Bars.
COMPLETED
Chapter 2
Volume Bars outperformed on Stability (lower CV) and Normality (lower JB), despite Dollar Bars being theoretically preferred for equities. Serial correlation differences were negligible.
0.7 Strategic Pivot
Lock in Volume Bars (VB) using empirically reasonable initial thresholds.
CONFIRMED
Chapter 2
The statistical edge is sufficient to defer further threshold optimization; the true signal is likely downstream in feature engineering.

--------------------------------------------------------------------------------
Immediate Next Steps (Phase 1: Labeling and Baseline Check on BTC)
Step
Action/Description
Key Objective
Relevant Chapter(s)
1. Define Dynamic Thresholds
Calculate EWMSD of VB returns (span=20 bars). Test multipliers: 1σ, 1.5σ, 2σ. Pick one that gives ∼40−60% label balance. Lock it in.
Establish dynamic profit-taking and stop-loss limits (τ) that adjust for volatility, a requisite for realistic labeling.
Chapter 3 (Computing Dynamic Thresholds)
2. Implement Triple-Barrier Labels
Code the TBM logic (upper, lower, vertical barriers). Apply to BTC first. Generate label distribution (% of −1,0,1). Sanity check: manual spot-check 20 labels for look-ahead bias.
Create robust, path-dependent prediction targets (y) that accurately reflect market exit mechanics (stop-loss/profit-taking).
Chapter 3 (The Triple-Barrier Method)
3. Run Baseline Classifier
Split data 70/30 (time-ordered). Use Logistic Regression on VB OHLCV → labels. Report accuracy on test set. Record if ≥52% or <52%.
Conduct the essential viability check. If accuracy is ≥52%, proceed confidently with feature engineering; if <52%, confirm that the raw sampling layer is insufficient.
Chapter 7 (Cross-Validation in Finance) & Part 2: Modelling

--------------------------------------------------------------------------------
