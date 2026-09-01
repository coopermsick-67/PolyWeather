# PolyWeather daily Tmax model card

## Decision

**SHADOW_ONLY** - The XGBoost candidate cleared the rolling-backtest comparison gates, but remains shadow-only until forecasts are prospectively logged from one frozen issue-time contract and re-verified.

## What was predicted

Official daily maximum temperature (TMAX, °F) at KNYC, KMIA, KMDW, KLAX, and KSFO. Labels are NOAA/NCEI daily-summaries records, not maxima reconstructed from rounded METAR observations.

## Method

The shadow candidate preserves the archived NCEP NBM hourly temperature curve as its baseline and uses gradient-boosted residual learning over NBM, HRRR, and GFS forecast profiles plus station ID and seasonal features. The backtest uses contiguous, calendar-ordered 31-day folds; no target date is scored twice, no random split or future reanalysis field is used.

## Held-out result

Across all rolling held-out station-days, NBM MAE was **2.65°F** and the XGBoost residual model MAE was **1.78°F**. Relative MAE skill was **32.8%**.

| station | NBM MAE °F | XGBoost MAE °F | XGBoost bias °F | XGBoost skill vs NBM | n |
|---|---:|---:|---:|---:|---:|
| KLAX | 3.62 | 1.65 | -0.09 | 54.3% | 492 |
| KMDW | 2.28 | 1.92 | +0.30 | 15.8% | 492 |
| KMIA | 2.44 | 1.15 | +0.03 | 53.0% | 491 |
| KNYC | 2.16 | 1.85 | +0.36 | 14.3% | 492 |
| KSFO | 2.76 | 2.34 | +0.60 | 15.0% | 492 |

A date-block bootstrap 95% interval for the XGBoost MAE is **1.72-1.85°F**; its skill interval versus NBM is **30.5%-35.1%**.

## Critical caveats

- This is a fixed-lead hourly reconstruction: each input hourly value was archived at a 24-hour lead, then aggregated to local-day Tmax. It is leakage-resistant, but it is not a reconstruction of one frozen once-per-day NWS forecast issuance.
- The current model uses the public archived NCEP NBM/HRRR/GFS forecast feed as its numerical guidance source. It does not yet include a historical immutable NWS forecastGridData archive, raw NBM GRIB neighborhood features, real-time ASOS histories, radar, satellite, or SST.
- The reported score is specific to the tested period, stations, target definition, lead convention, sources, and their model versions. It is not a promise for future daily highs; keep the model in shadow monitoring through seasonal transitions and upstream NWP changes.
- Prediction intervals are split-conformal, nominal 75% bands calibrated using prior dates only. Monitor empirical coverage after deployment.

## Reproducibility

Artifacts include the feature table checksum, row-level rolling predictions, per-station metrics, date-block bootstrap confidence intervals, acceptance decision, serialized model, and run manifest.
