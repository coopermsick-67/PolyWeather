# Accuracy experiments and completed correctness fixes

**85% all-city within-2°F accuracy and an 85% operational betting win rate have not been achieved.** The software fixes and experiments are complete; source collection, deployment, market integration, and prospective validation remain outstanding.

This replaces the earlier interpretation of the threshold experiment. A historical hit rate on a hypothetical line does not establish that the line is offered by a market, that its estimated probability is calibrated, or that trading it is profitable. An unsuccessful candidate search also does not establish a theoretical skill ceiling.

## Reproduced results

The original rolling experiment remains 67.886% within 2°F across 10,279 rows. The new comparison trained/calibrated before its validation and test periods, excluding 60 inadequate training profiles. Validation covers May 5–July 3, 2026 (1,200 rows); test covers July 4–September 1 (1,199 usable rows). These are previously inspected historical data with an issuance convention that differs from live requests. The chronological test is not fresh prospective evidence.

| Evaluation on the 1,199-row test | Rate | Descriptive 95% seven-date-block interval |
|---|---:|---:|
| All-city point forecast within 2°F | 68.31% | 66.25–70.37% |
| Default even-anchored two-degree bucket | 38.03% | 34.81–41.17% |
| Hypothetical lower threshold | 91.83% | 90.42–93.08% |
| Hypothetical upper threshold | 93.16% | 91.83–94.65% |
| Verified operational/financial win rate | Unavailable | Unavailable |

The two challenger regressors scored 61.42% and 61.83% within 2°F on validation, versus 65.58% for the incumbent approach. They were rejected before testing. No production artifact was replaced or deployed. The incumbent refit reached 1.715°F test MAE. This split differs from the original rolling evaluation; the difference in rates is not evidence of an operational improvement.

The station-only selection policy chose Las Vegas and Phoenix using validation alone. It reached 70.83% within 2°F on 120 test rows, at 10.01% selection coverage, with a 60.00–80.83% block interval. It failed the 85% target.

## Corrected threshold and selection experiment

The confidence regressor now fits on the first 30 validation dates and selects its cutoff on the last 30 dates. Previously it selected cutoffs on the same rows used to fit confidence. No tested cutoff reaches 85% at 5% minimum coverage: best observed selection-window rates are 76.32% within 2°F and 48.32% exact-bucket hits. These describe this particular search, not the best possible model or selector.

The threshold experiment now uses explicit integer lines matching the probability helper's nearest-degree convention:

```
settled = floor(actual + 0.5)
lower_line = floor(prediction - margin + 0.5)
upper_line = ceil(prediction + margin - 0.5)
lower_hit = settled >= lower_line
upper_hit = settled <= upper_line
```

It selects the smallest margin meeting 85% on both sides in validation and scores only that margin on test. The corrected margin is 2.5°F before outward rounding. With whole-degree labels this produces the same test hit rates as the previous unrounded ±3°F comparison. All lines are hypothetical; 100% evaluation coverage is not 100% executable-market coverage. This rerun corrects the prior analysis on already inspected test data and is not a newly untouched test.

The threshold helper now handles empirical atoms at settlement boundaries, validates integer lines, returns the most demanding line satisfying the requested probability, and bounds its search in both directions. Its output is still only an estimate conditional on the supplied distribution. It does not generate live recommendations or replace the decision gates.

## Correctness fixes and validation

Across this work and the preserved preceding fixes:

- Missing, boolean, nonnumeric, and nonfinite source values cannot masquerade as valid temperatures or coverage. Missing availability fails closed. Expected full local-day hours, including DST, determine coverage; truncated and duplicated timestamps cannot inflate it. Malformed hourly arrays fail explicitly.
- NCEI labels with quality flags or nonfinite temperatures are excluded. Retraining from older tables applies profile checks. Missing/fractional/boolean horizons and duplicate training station-dates are rejected.
- Empirical buckets use the same half-open settlement windows as the result scorer. Observed-floor conditioning retains mass exactly at the floor and applies to quantiles. Bucket grids remain aligned for arbitrary widths.
- Unknown/invalid run age and unverified probability calibration block recommendations. Retrieval time is not source-run age. Unknown revision history blocks recommendations; refresh padding cannot dilute movement per hour, and future snapshots cannot provide evidence.
- Display smoothing no longer translates the calibrated interval. The displayed point still uses a continuity policy whose accuracy has not been independently replayed.
- Prospective logging serializes duplicate checks and appends across processes, timestamps predictions after source retrieval, and records source failures per station. Verification rejects nonprospective timestamps, duplicate or corrupted records, uses station-local maturity, and reports outages and unresolved labels.
- Existing browser/Worker fallbacks continue clearing stale recommendation evidence and rejecting invalid temperatures.

Verified using the workspace Python 3.11 environment: **206 Python tests passed**, including 73 regressions added after the preceding 133-test state. **24 JavaScript tests passed**, `npm run build` passed, Worker syntax passed, and `git diff --check` passed. Tests verify behavior, not predictive accuracy. Global Python 3.14 crashed inside pandas during the experiment; use `.venv/Scripts/python.exe` for reproducibility.

## Artifacts and reproduction

```
.venv/Scripts/python.exe scripts/evaluate_accuracy_candidates.py
.venv/Scripts/python.exe scripts/evaluate_selective_win_rate.py --output reports/accuracy_improvement_2026-09-05/corrected
.venv/Scripts/python.exe -m pytest
```

The candidate run saves `protocol.json`, `frozen_selection.json`, `summary.json`, and both prediction Parquets in this directory. The corrected analysis is in `corrected/selective_win_rate.json`, including script and prediction-file hashes. Previous JSON is retained for comparison. Generated JSON/Parquet artifacts are locally available but ignored by the repository's existing artifact rules; reproduce them with the commands above.

Remaining prerequisites for a credible 85% operational claim: independently identified forecast runs; matching historical/live cutoffs; verified provider contracts and executable lines/prices; appropriately calibrated conditional distributions; successful serving of the intended model; and an immutable, resolved prospective record with adequate dates, selection coverage, and uncertainty. No future hit rate, universal skill ceiling, or financial edge is inferred from the historical threshold experiment.
