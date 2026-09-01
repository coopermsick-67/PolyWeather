# Model upgrade validation (August 2026)

## Decision

The versioned XGBoost residual candidate in `artifacts/production_v2` replaces the earlier local dashboard candidate for shadow monitoring. It remains `SHADOW_ONLY`: this is a statistically supported retrospective improvement, not a guarantee of future or operational accuracy.

## What changed

The model retains NBM local-day hourly Tmax as its physical baseline and learns a station-aware residual. The updated feature contract adds only information available in the forecast inputs:

- cross-model agreement, spread, and standard deviation for daily maximum, minimum, and mean temperature;
- NBM-minus-HRRR and NBM-minus-GFS temperature differences; and
- early-versus-late forecast-profile means and warming signals for each source.

No observation after the forecast target time, reanalysis field, or target-derived feature was added.

## Strict rolling result

The available complete archive covers 2024-10-08 through 2026-08-10. The expanding-origin evaluation contains 16 contiguous test folds, 2,459 unique station-date forecasts, and 492 unique target dates from 2025-04-06 through 2026-08-10.

| Candidate | MAE (degrees F) | RMSE | Bias | Within 2 degrees F | P90 absolute error | Skill vs. NBM |
|---|---:|---:|---:|---:|---:|---:|
| Raw NBM | 2.651 | 3.309 | -1.909 | 45.2% | 5.300 | -- |
| Earlier XGBoost | 1.825 | -- | -- | -- | -- | 31.2% |
| Updated XGBoost | **1.783** | **2.421** | +0.243 | **66.6%** | **3.732** | **32.8%** |

The new model reduced held-out MAE by 0.042 degrees F versus the earlier XGBoost. A paired bootstrap that resamples whole target dates gives a 95% interval of **0.011 to 0.071 degrees F** for that improvement, so the gain is positive but deliberately described as incremental.

The updated XGBoost beats raw NBM at all five stations. It improves the earlier XGBoost result at KLAX, KMDW, KMIA, and KNYC; KSFO is essentially unchanged and slightly worse by 0.006 degrees F. The main residual risk remains coastal/marine-layer days and lake-breeze or frontal days.

## Why this is not a claimed multi-year operational result

The exact archived multi-model feature contract becomes complete only on 2024-10-08. Earlier dates return missing NBM predictor values, so extending this score backward would silently change the input contract or require an entirely separate direct-NOAA-GRIB reconstruction. That work is worthwhile, but it must be run as a new, fixed-issuance dataset rather than mixed into this result.

The current historical evaluation is also a 24-hour hour-wise composite, not a single forecast run frozen at one daily cutoff. The model therefore stays in shadow monitoring until prospective fixed-cutoff forecasts are logged and scored against mature NOAA/NCEI TMAX labels.

## Evidence

- `artifacts/backtest_v2/overall_metrics.csv`
- `artifacts/backtest_v2/station_metrics.csv`
- `artifacts/backtest_v2/block_bootstrap_ci.csv`
- `artifacts/backtest_v2/acceptance.json`
- `artifacts/production_v2/model_manifest.json`
- `reports_v2/MODEL_CARD.md`
