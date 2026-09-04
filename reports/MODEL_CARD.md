# PolyWeather daily Tmax model card

## Decision

**SHADOW_ONLY** - The XGBoost candidate cleared the rolling-backtest comparison gates, but remains shadow-only until forecasts are prospectively logged from one frozen issue-time contract and re-verified.

## What was predicted

Official daily maximum temperature (TMAX, °F) at the 20 settlement stations declared in the model manifest. Labels are NOAA/NCEI daily-summaries records, not maxima reconstructed from rounded METAR observations.

## Method

The shadow candidate preserves the archived NCEP NBM hourly temperature curve as its baseline and uses gradient-boosted residual learning over NBM, HRRR, and GFS forecast profiles plus station ID and seasonal features. The backtest uses contiguous, calendar-ordered 31-day folds; no target date is scored twice, no random split or future reanalysis field is used.

## Held-out result

Across all rolling held-out station-days, NBM MAE was **2.41°F** and the XGBoost residual model MAE was **1.75°F**. Relative MAE skill was **27.4%**.

| station | NBM MAE °F | XGBoost MAE °F | XGBoost bias °F | XGBoost skill vs NBM | n |
|---|---:|---:|---:|---:|---:|
| KATL | 2.91 | 1.78 | +0.09 | 38.7% | 514 |
| KAUS | 2.60 | 1.87 | +0.25 | 28.1% | 514 |
| KBOS | 2.40 | 1.90 | +0.14 | 20.6% | 514 |
| KDCA | 2.59 | 2.11 | +0.41 | 18.4% | 514 |
| KDEN | 2.57 | 2.05 | +0.44 | 20.3% | 514 |
| KDFW | 2.06 | 1.89 | +0.29 | 8.2% | 514 |
| KHOU | 2.86 | 1.54 | +0.39 | 46.0% | 514 |
| KLAS | 1.44 | 1.18 | +0.05 | 18.1% | 514 |
| KLAX | 3.69 | 1.64 | -0.14 | 55.6% | 514 |
| KMDW | 2.27 | 1.89 | +0.18 | 16.8% | 514 |
| KMIA | 2.45 | 1.16 | -0.02 | 52.5% | 513 |
| KMSP | 2.29 | 1.84 | +0.23 | 19.3% | 514 |
| KMSY | 2.21 | 1.57 | +0.28 | 28.8% | 514 |
| KNYC | 2.14 | 1.74 | +0.25 | 18.7% | 514 |
| KOKC | 2.34 | 1.90 | +0.18 | 18.9% | 514 |
| KPHL | 2.62 | 1.76 | +0.12 | 33.0% | 514 |
| KPHX | 1.89 | 1.24 | +0.09 | 34.5% | 514 |
| KSAT | 1.93 | 1.69 | +0.23 | 12.0% | 514 |
| KSEA | 2.22 | 1.86 | +0.42 | 16.2% | 514 |
| KSFO | 2.73 | 2.37 | +0.62 | 13.2% | 514 |

A date-block bootstrap 95% interval for the XGBoost MAE is **1.72-1.79°F**; its skill interval versus NBM is **26.0%-28.8%**.

## Critical caveats

- This is a fixed-lead hourly reconstruction: each input hourly value was archived at a 24-hour lead, then aggregated to local-day Tmax. It is leakage-resistant, but it is not a reconstruction of one frozen once-per-day NWS forecast issuance.
- The current model uses the public archived NCEP NBM/HRRR/GFS forecast feed as its numerical guidance source. It does not yet include a historical immutable NWS forecastGridData archive, raw NBM GRIB neighborhood features, real-time ASOS histories, radar, satellite, or SST.
- The reported score is specific to the tested period, stations, target definition, lead convention, sources, and their model versions. It is not a promise for future daily highs; keep the model in shadow monitoring through seasonal transitions and upstream NWP changes.
- Prediction intervals are asymmetric split-conformal, nominal 80% bands calibrated using prior dates only with conservative finite-sample order statistics. Monitor empirical coverage after deployment.

## Reproducibility

Artifacts include the feature table checksum, row-level rolling predictions, per-station metrics, date-block bootstrap confidence intervals, acceptance decision, serialized model, and run manifest.
