# Hyper-analysis of the supplied temperature-prediction report

## Bottom-line assessment

The supplied report is a strong **system-design brief**, not evidence that a model has already been trained or that any stated accuracy has been achieved. Its best ideas are the right ones for this problem: use an official station-local daily maximum as truth, preserve a numerical-weather baseline, learn station-specific residual error rather than replace meteorology with a black box, and validate in time order. This project implements those principles.

The report's biggest unresolved issue is operational, not algorithmic: a daily forecast must be defined by both its target local day and the exact information cutoff. Without a frozen issue time, source run, valid-time window, and ingestion record, an apparently excellent historical score can be accidentally optimistic. The present implementation therefore has a deliberately conservative `SHADOW_ONLY` release decision.

## What was adopted, what was narrowed, and what remains open

| Report premise | Implementation decision | Status | Why it matters |
|---|---|---|---|
| Forecast daily high temperature for five airport/city stations | Target NOAA/NCEI Daily Summaries `TMAX` on each station's local calendar day | Implemented | A target must be exact before any score can mean anything. |
| Use NBM/NWP guidance instead of pure historical-temperature extrapolation | Raw NBM local-day hourly Tmax is the baseline; HRRR and GFS profiles are auxiliary inputs | Implemented | The residual model corrects a physically informed forecast rather than relearning weather from a small station dataset. |
| Correct local bias and microclimate effects | Pooled station-aware Ridge and XGBoost residual MOS models | Implemented | The model can learn persistent station and regime differences while sharing information across sites. |
| Compare simple baselines before complex models | Seasonal climatology, raw NBM, Ridge residual, then XGBoost residual | Implemented | This establishes whether additional complexity creates material incremental value. |
| Use historical forecast vintages, not outcome-aware fields | Open-Meteo `*_previous_day1` archived fields | Partly implemented | They are fixed-lead and safer than reanalysis, but form an hour-wise composite rather than one issuance. |
| Use official NWS forecasts and NBM | Current code uses NBM-labelled numerical guidance; it does not ingest a historical immutable NWS forecast-grid archive | Not yet implemented | NWS point/grid forecasts and NBM are related but not interchangeable products. |
| Rolling temporal validation and deployment gates | Clean expanding-origin folds, date-block bootstrap, append-only shadow logging | Implemented, with an operational blocker | Retrospective comparison is necessary but not sufficient for release. |

## What the model actually estimates

For station `s` and local target day `d`, the baseline is the maximum of the archived NBM 2 m temperature profile over the local day:

```text
b(s,d) = max[NBM temperature at local-day target hours]
forecast(s,d) = b(s,d) + residual_model(features(s,d))
```

The feature set contains hourly-profile summaries from NBM, HRRR, and GFS (temperature, dew point, humidity, cloud, wind, precipitation, radiation, and pressure when available), plus station identity and calendar variables. The XGBoost model is intentionally modest: shallow trees, regularization, subsampling, and a 60-day trailing calibration block. It is not a deep-learning model, and that is appropriate for the present data volume and the need to preserve a clear NBM anchor.

The calibration block supplies station-level median residual offsets and nominal 80% split-conformal intervals. It is earlier than the test/future period, so it does not reuse later outcomes to calibrate those forecasts.

## Evidence from the corrected backtest

The corrected evaluation contains 16 contiguous expanding-origin folds covering 2025-04-06 through 2026-08-10. It has **2,459 unique station-date forecasts**, **492 unique target dates**, and **zero repeated station-date rows**. A date-block bootstrap resamples whole target dates, retaining the correlation among the five same-day stations rather than pretending they are independent.

| Model | MAE (°F) | RMSE (°F) | Bias (°F) | P90 absolute error (°F) | MAE skill vs. raw NBM |
|---|---:|---:|---:|---:|---:|
| Seasonal climatology | 6.28 | 8.86 | -1.70 | 14.42 | -136.9% |
| Raw NBM local-day Tmax | 2.65 | 3.31 | -1.91 | 5.30 | 0.0% |
| Ridge residual | 1.88 | 2.53 | +0.19 | 4.07 | 29.2% |
| XGBoost residual | **1.82** | **2.44** | **+0.17** | **3.87** | **31.2%** |

The XGBoost estimate has a date-block bootstrap 95% MAE interval of **1.76 to 1.89°F** and a skill interval of **29.0% to 33.2%** versus raw NBM. Its nominal 80% interval covered 86.5% of held-out outcomes with a mean width of 7.09°F. This is mildly conservative on this historical evaluation; it does not establish future calibration.

The model leads raw NBM at every station, but it does not lead the Ridge residual model at every station. That distinction matters: XGBoost wins overall by only 0.05°F MAE, while Ridge is better at KLAX and KSFO. The selected model is a reasonable general candidate, not evidence that a single nonlinear model dominates every local regime.

| Station | Raw NBM MAE | Ridge MAE | XGBoost MAE | XGBoost skill vs. NBM | XGBoost interval coverage |
|---|---:|---:|---:|---:|---:|
| KLAX | 3.62 | **1.61** | 1.70 | 53.1% | 88.0% |
| KMDW | 2.28 | 2.10 | **2.05** | 9.8% | 87.6% |
| KMIA | 2.44 | 1.34 | **1.16** | 52.6% | 87.0% |
| KNYC | 2.16 | 2.00 | **1.87** | 13.2% | 86.4% |
| KSFO | 2.76 | **2.33** | 2.34 | 15.2% | 83.5% |

There is no causal conclusion in this table about why a station improves more than another. It does show that a global score should not hide station-level behavior: the large KLAX and KMIA improvements should not be read as an equivalent operational advantage at KMDW, KNYC, or KSFO.

## Data audit

The training table has 3,359 station-date rows from 2024-10-08 through 2026-08-10. Its grain is one station by one local target date by one archived 24-hour lead. The audit found:

- Zero duplicate station-date rows.
- No missing `TMAX` labels or NBM baseline values.
- Plausible `TMAX` values from 2.0 to 100.0°F and NBM baselines from 3.6 to 101.5°F.
- Five rows with less than 90% NBM hourly-profile availability.
- Some pressure and shortwave-radiation fields fully absent in this public archive; the feature selector excludes fully absent or invariant fields rather than fabricating values.

This is a good basic data-quality result, but it is not a proof of operational source integrity. The main remaining issue is that the archive's `previous_day1` convention supplies an individual 24-hour-old forecast for every valid hour. Aggregating those hours into one local-day maximum means the final daily profile contains different issuance times.

## The key accuracy limitation

The backtest says the candidate improves a **24-hour-lead hourly composite**. It does not yet say that it improves a forecast made, for example, at one fixed local time on the prior day. The live prediction path currently asks for the latest available multi-model forecast at request time, which is not the same frozen-vintage contract as the historical composite.

That gap prevents a strong operational claim even though the retrospective score is promising. It also guards against a common but subtle leakage failure: silently using a later forecast run for some target hours than would have been available at the chosen issuance cutoff.

Other limits worth keeping visible:

1. The archived sample is relatively short and may underrepresent rare heat, cold, tropical, marine-layer, and frontal regimes.
2. NBM versions, gridding, and downstream provider processing can change over time. A pooled model may learn an era-specific bias.
3. The model family and hyperparameters were compared on the same rolling evaluation used for reporting. That is reasonable development evidence, but not an untouched final selection test.
4. NCEI labels can be delayed or revised. Prospective scoring must keep the source snapshot and verify labels only when mature.
5. Coastal sites are sensitive to grid-cell and elevation-selection policy. That policy must be versioned if raw gridded data are used.
6. Empirical interval coverage is good on the current test set, but interval sharpness and coverage must be tracked separately by station and season.

## Recommended hardening path

The most valuable next work is not a larger neural network. It is a stricter data-vintage pipeline:

1. **Write the forecast contract.** Specify target local-day boundaries, a single cutoff in UTC and local time, an allowed publication-lag rule, and a deterministic source fallback.
2. **Use immutable run-level data.** Retrieve the matching NOAA NBM GRIB objects (or an equivalently immutable source) using the recorded run initialization and valid times. Save source URL, object version/last-modified time, checksum, grid-cell policy, and feature extraction version.
3. **Backtest exactly the live contract.** Rebuild history using only runs available before each cutoff. Do not mix per-hour issue times into a daily forecast once the fixed-cutoff product is defined.
4. **Run prospective shadow monitoring.** Append predictions before the target day, retain input snapshots, then join only to mature NCEI `TMAX` labels. Compare paired absolute errors to raw NBM by station and season.
5. **Predeclare promotion and rollback criteria.** The current statistical gate is more than 1% global MAE skill and at least four station wins versus NBM; it passed five. Keep the separate issuance-integrity gate, and fall back to raw NBM whenever source completeness or timing fails.

## Verdict

The project faithfully implements the report's most defensible core: NBM as a baseline, station-aware residual learning, official labels, and calendar-ordered validation. The result is strong enough to justify continued shadow operation and a strict vintage-data upgrade. It is not yet strong enough to promise a deployed, single-cutoff daily-high product, because that product has not been reconstructed and scored under an identical information-set contract.

## Evidence files

- `artifacts/quality/TRAINING_DATA_QUALITY.md` - source-table QA.
- `artifacts/backtest/rolling_predictions.parquet` - row-level rolling-origin predictions.
- `artifacts/backtest/overall_metrics.csv` - model-level scores.
- `artifacts/backtest/station_metrics.csv` - station diagnostics and interval coverage.
- `artifacts/backtest/block_bootstrap_ci.csv` - date-block bootstrap intervals.
- `artifacts/backtest/acceptance.json` - the explicit `SHADOW_ONLY` decision and release blocker.
- `artifacts/production/model_manifest.json` - final training and model metadata.

## Primary source families

- [NOAA/NCEI Daily Summaries service](https://www.ncei.noaa.gov/access/services/data/v1)
- [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
- [NOAA National Blend of Models open-data registry](https://registry.opendata.aws/noaa-nbm/)
- [NOAA NBM download documentation](https://vlab.noaa.gov/web/mdl/nbm-download)
