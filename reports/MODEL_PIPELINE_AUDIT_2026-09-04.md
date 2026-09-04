# WeatherPicks model and prediction-pipeline audit

Date: 2026-09-04

## Bottom line

The residual XGBoost model has real retrospective value, but the previous application contract and uncertainty reporting were unsafe. The v4 rolling evaluation covers 10,279 unique held-out station-days and records 1.750°F MAE versus 2.411°F for raw NBM, or 27.4% relative MAE skill. All 20 stations improve over NBM. The model remains `SHADOW_ONLY` because the archived inputs are hour-wise 24-hour-lead composites rather than one immutable forecast issuance.

## What was already good

- The target is official NOAA/NCEI station-local daily TMAX, not a city-center proxy or a reconstructed UTC-day maximum.
- The prediction is anchored to a strong physical NBM forecast and learns only its residual.
- NBM, HRRR, and GFS profiles, station identity, calendar seasonality, model disagreement, and diurnal-shape features are available to the learner.
- Rolling-origin folds are chronological and non-overlapping; no future target row enters a fold's training data.
- Date-block bootstrap intervals preserve same-day cross-station dependence better than row-wise resampling.
- The production candidate is kept in shadow mode pending prospective fixed-cutoff evidence.

## Material flaws found and remediation

| Severity | Finding | Risk | Remediation in v4 |
|---|---|---|---|
| Critical | A one-day residual model was applied to today and days 2-7. | Unsupported horizons looked calibrated and inherited one-day accuracy. | Residual correction now requires `forecast_lead_days == 1`; other dates use live numerical guidance and are labeled uncalibrated. |
| Critical | Fallbacks manufactured ±3°F/±4°F bands and could display packaged temperatures after all live sources failed. | Plausible-looking random ranges or stale values could be mistaken for predictions. | Uncalibrated fallbacks expose no interval; total live-source failure returns an unavailable state with no cached temperature substitution. |
| High | Public `p10/p90` fields advertised a 75% interval, and the 20-station artifact covered only 71.3%. | The interval semantics were false and systematically overconfident. | Nominal coverage is now 80%; asymmetric residual tails use outward finite-sample order statistics plus a prior-only drift check. Canonical v4 coverage is 84.6%, with every station above 75%. |
| High | Model acceptance ignored interval coverage and the bootstrap result. | A point model could pass while uncertainty failed or apparent skill was not robust. | Acceptance now gates global coverage, minimum station coverage, all-station NBM wins, Ridge comparison, and positive date-block bootstrap skill. |
| High | XGBoost versus blend could be chosen after observing the same reported test folds. | Post-hoc winner selection biases the final score. | XGBoost is predeclared as the candidate; blend remains a diagnostic challenger with its own separately scored interval. |
| High | Complete core highs were checked, but broad feature loss could still be median-imputed silently. | A degraded upstream payload could produce a confident but out-of-contract residual. | Live residual inference and shadow logging require at least 85% finite expected numeric features. |
| Medium | Training accepted mixed lead horizons if supplied. | One estimator could learn an incoherent blend of different forecasting tasks. | Training rejects any horizon set other than exactly one day and instructs operators to train separate artifacts. |
| Medium | Data quality reported only NBM availability and omitted horizon, core-source, gap, and all-null feature diagnostics. | Source regressions could hide inside a successful build. | The audit now reports all three model availabilities, core guidance gaps, station calendar gaps, horizon values, per-day station counts, hash cardinality, and all-null numeric fields. |
| Medium | Serialized XGBoost artifacts did not record runtime versions. | A pickle created under another XGBoost/sklearn version can warn, fail, or behave incompatibly. | New manifests record Python, NumPy, pandas, scikit-learn, and XGBoost versions plus the artifact format. A v4 artifact was retrained under those pinned versions. |

## v4 data-quality evidence

- 13,879 rows from 2024-10-08 through 2026-09-01.
- 20 configured stations; 694 days each except KMIA with one missing official day.
- Zero duplicate station/date keys.
- Zero missing TMAX, NBM baseline, or NBM/HRRR/GFS daily-high inputs.
- Exactly one forecast horizon is present: one day.
- Six NBM archive fields are 100% unavailable and are automatically excluded; they are not filled with invented weather.
- No implausible TMAX or NBM values under the current physical bounds.

## v4 rolling evidence

| Model | MAE °F | RMSE °F | Bias °F | Within 2°F | Skill vs NBM | Interval coverage | Mean interval width °F |
|---|---:|---:|---:|---:|---:|---:|---:|
| NBM | 2.411 | 3.038 | -1.507 | 50.3% | 0.0% | — | — |
| Ridge residual | 1.880 | 2.518 | +0.152 | 63.4% | 22.0% | 84.4% | 7.02 |
| XGBoost residual | **1.750** | **2.399** | +0.225 | **67.9%** | **27.4%** | **84.6%** | **6.65** |
| Station blend | 1.757 | 2.395 | +0.204 | 67.5% | 27.1% | 87.9% | 7.46 |

The XGBoost date-block bootstrap 95% interval is 1.717-1.787°F for MAE and 26.0%-28.8% for relative MAE skill. Station skill ranges from 8.2% at KDFW to 55.6% at KLAX. Station interval coverage ranges from 78.6% at KSFO to 87.4% at KMSP.

## Remaining blockers

1. The archived `previous_day1` features are a per-valid-hour 24-hour-lead composite, not one forecast run frozen at a declared daily cutoff.
2. The live Open-Meteo endpoint does not expose a stable underlying model-run identifier, so exact historical/live feature parity is not proven.
3. No amount of retrospective retuning removes those provenance gaps. Promotion requires prospective forecasts logged at one fixed cutoff, immutable feature snapshots and model hashes, mature NCEI labels, and the same station/coverage/bootstrap gates over seasonal transitions.
4. Multi-day residual skill is unknown. Separate lead-2 through lead-7 datasets, models, calibration, and acceptance evidence are required before those horizons may be called calibrated.

The safe current behavior is therefore: v4 residual MOS for a fully populated one-day forecast only; real NBM/NWS point guidance without a manufactured interval elsewhere; no output when all live sources fail.
