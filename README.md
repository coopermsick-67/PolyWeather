# PolyWeather: daily high-temperature forecasting

PolyWeather is a reproducible, leakage-aware research system for forecasting the official daily high temperature at KNYC, KMIA, KMDW, KLAX, and KSFO.

**Current release status: `SHADOW_ONLY`.** The residual model materially improved the tested 24-hour-lead composite, but it has not yet passed prospective validation from one frozen operational issue-time contract. It is a useful forecasting candidate, not a promise that every future high will be accurate.

## What the system predicts

| ICAO | Location | NCEI station ID | Local target-day timezone |
|---|---|---|---|
| KNYC | Central Park, NY | USW00094728 | America/New_York |
| KMIA | Miami International Airport, FL | USW00012839 | America/New_York |
| KMDW | Chicago Midway Airport, IL | USW00014819 | America/Chicago |
| KLAX | Los Angeles International Airport, CA | USW00023174 | America/Los_Angeles |
| KSFO | San Francisco International Airport, CA | USW00023234 | America/Los_Angeles |

The label is `TMAX` from NOAA/NCEI Daily Summaries in degrees Fahrenheit: the official maximum for each station's local calendar day. It is not a maximum reconstructed from rounded METAR observations and it is not a UTC-day maximum.

## Forecast contract and model

The historical experiment uses Open-Meteo Previous Runs fields named `*_previous_day1` for NCEP NBM CONUS, HRRR CONUS, and GFS Seamless. Each hourly value is archived at a fixed 24-hour lead; the system converts the hourly profiles to the station's local day and takes the NBM hourly-temperature maximum as the raw baseline.

The original production candidate is a residual MOS model:

```text
daily-TMAX forecast = NBM local-day hourly Tmax + learned residual
```

The residual uses NBM, HRRR, and GFS hourly-profile summaries, station identity, and calendar features. A Ridge residual model is the transparent comparison model; a conservatively regularized XGBoost residual model is the selected candidate. Its prediction bands use a trailing, time-ordered 60-day calibration block and split-conformal residual widths.

The dashboard uses the versioned XGBoost residual candidate in `artifacts/production_v2`. On the 2,459 untouched rolling forecasts its MAE was **1.783°F**, versus **2.651°F** for raw NBM. This remains a shadow-only result, not a guarantee of future accuracy or multi-day skill.

Important: a `previous_day1` daily maximum is an **hour-wise 24-hour-lead composite**, not the output of one forecast issuance frozen at a single clock time. It is leakage-resistant with respect to the outcome, but it is not yet the same contract as a once-daily operational forecast. That distinction is why the release is shadow-only.

## Clean rolling-origin result

The final evaluation uses 16 contiguous, expanding-origin test folds. Each fold is 31 days except the final partial fold; all training rows predate their fold, and no station-date occurs in more than one held-out fold. The resulting sample contains **2,459 unique station-date forecasts** across **492 unique local dates**, from 2025-04-06 through 2026-08-10.

| Model | MAE (°F) | RMSE (°F) | Bias (°F) | Within 2°F | MAE skill vs. NBM |
|---|---:|---:|---:|---:|---:|
| Seasonal climatology | 6.28 | 8.86 | -1.70 | 26.0% | -136.9% |
| Raw NBM local-day Tmax | 2.65 | 3.31 | -1.91 | 45.2% | 0.0% |
| Ridge residual | 1.88 | 2.53 | +0.19 | 64.9% | 29.2% |
| XGBoost residual | **1.82** | **2.44** | **+0.17** | 64.0% | **31.2%** |

The date-block bootstrap 95% interval for XGBoost MAE is **1.76 to 1.89°F**; its MAE-skill interval versus raw NBM is **29.0% to 33.2%**. The nominal 75% prediction interval covered 71.2% of held-out outcomes, with a mean width of 4.48°F. A fixed ±2°F (four-degree) planning range contained 66.6% of held-out outcomes. These are retrospective results for this exact target, data vintage, lead convention, and station set.

| Station | Raw NBM MAE | XGBoost MAE | XGBoost bias | XGBoost skill | Held-out n |
|---|---:|---:|---:|---:|---:|
| KLAX | 3.62 | 1.70 | -0.09 | 53.1% | 492 |
| KMDW | 2.28 | 2.05 | +0.14 | 9.8% | 492 |
| KMIA | 2.44 | 1.16 | +0.06 | 52.6% | 491 |
| KNYC | 2.16 | 1.87 | +0.31 | 13.2% | 492 |
| KSFO | 2.76 | 2.34 | +0.44 | 15.2% | 492 |

The model improved over raw NBM at all five stations, but the size of the improvement is uneven. KLAX and KMIA drive much of the gain; KMDW, KNYC, and KSFO have smaller margins. That heterogeneity is reason to monitor station-level performance rather than deploy on the global MAE alone.

## Data quality and lineage

The current training table has 3,359 station-date rows, covering 2024-10-08 through 2026-08-10. Quality checks found zero duplicate station-date rows, no missing target or NBM-baseline values, labels in a plausible 2.0 to 100.0°F range, and NBM baselines in a plausible 3.6 to 101.5°F range. Five rows have less than 90% NBM hourly-profile availability. Fields that are entirely unavailable in this archived feed are excluded automatically rather than imputed as invented weather.

| Role | Source used now | Why it matters |
|---|---|---|
| Truth label | NOAA/NCEI Daily Summaries `TMAX` | Keeps the target official and station-specific. |
| Main numerical baseline | NCEP NBM CONUS, accessed through Open-Meteo Previous Runs | Provides the physical day-ahead temperature profile. |
| Complementary numerical features | NCEP HRRR CONUS and GFS Seamless, through the same archive | Lets the residual model respond to forecast-profile differences. |
| Audit trail | Parquet feature table, metadata hash, row-level rolling predictions, model manifest, shadow JSONL | Makes source and evaluation evidence inspectable. |

The current archive is not a substitute for a direct, immutable NOAA NBM GRIB run archive. It is deliberately documented as a rapid, testable composite workflow, not presented as an official NWS point-forecast system.

## Reproduce the current pipeline

Use Python 3.12 in the workspace virtual environment. The commands below are PowerShell commands.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e . pytest

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\.venv\Scripts\python.exe -m pytest
```

Build the labeled archived-feature table and run its quality audit:

```powershell
.\.venv\Scripts\python.exe -m polyweather.cli build-data `
  --start 2024-10-08 --end 2026-08-11 `
  --output data\features\tmax_24h_composite_training.parquet

.\.venv\Scripts\python.exe -m polyweather.cli quality `
  --data data\features\tmax_24h_composite_training.parquet `
  --output-dir artifacts\quality
```

Run the clean rolling evaluation, fit the candidate, and generate the model card and charts:

```powershell
.\.venv\Scripts\python.exe -m polyweather.cli backtest `
  --data data\features\tmax_24h_composite_training.parquet `
  --output-dir artifacts\backtest

.\.venv\Scripts\python.exe -m polyweather.cli train `
  --data data\features\tmax_24h_composite_training.parquet `
  --kind xgb --output-dir artifacts\production

.\.venv\Scripts\python.exe -m polyweather.cli report `
  --backtest-dir artifacts\backtest --output-dir reports
```

Generate a current forecast or begin the append-only shadow log:

```powershell
.\.venv\Scripts\python.exe -m polyweather.cli predict `
  --model artifacts\production\xgb_residual_tmax.joblib `
  --stations all

.\.venv\Scripts\python.exe -m polyweather.cli log-current `
  --model artifacts\production\xgb_residual_tmax.joblib `
  --stations all --log data\normalized\shadow_forecasts.jsonl

.\.venv\Scripts\python.exe -m polyweather.cli verify-shadow `
  --log data\normalized\shadow_forecasts.jsonl `
  --output artifacts\shadow\verified_forecasts.parquet
```

`verify-shadow` only scores forecasts whose target date has passed and for which NCEI labels can be retrieved. Preserve the JSONL file: the stored feature snapshot is part of the audit record.

## Local dashboard

The dashboard is a real React/Vite application. It calls the local trained-model adapter on refresh, shows a fixed ±2°F four-degree planning range (66.6% historical coverage on the untouched backtest), and exposes the wider per-station nominal-75% split-conformal range in station details. It includes the rolling shadow-monitoring chart and labels the product as experimental. It automatically rolls from today through the next seven calendar days (including in an open tab after midnight); today's estimate is never allowed below the latest observed NWS station temperature.

Forecasts are also display-stable: the latest model is recomputed on refresh, but a station/day’s displayed high remains unchanged for moves smaller than 2°F. A material 2°F-or-greater model shift, or a higher observed temperature today, updates it. This local continuity state is kept in `data\normalized\dashboard_forecast_state.json`.

```powershell
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Keep the workspace virtual environment and production model in place: the Vite development server invokes `scripts\dashboard_payload.py`, which loads `artifacts\production_v2\xgb_residual_tmax.joblib` and fetches the current forecast features.

## Deploying to the web (free)

The local dev server's `/api/dashboard` route only exists inside `vite dev` (it shells out to Python from a Vite middleware). For a real deployment, the same `payload()` logic is served by `server.py`, a small Flask app with no dev-only tricks, and the frontend calls it over `VITE_API_BASE_URL` instead of a relative path.

**Backend — Render (free web service)**

1. Push this repo to GitHub (see below).
2. On [render.com](https://render.com), New → Blueprint, point it at the repo. `render.yaml` at the repo root defines the service (`pip install -e ".[server]"`, `gunicorn server:app`, free plan) — Render reads it automatically.
3. Deploy. Note the resulting URL, e.g. `https://polyweather-api.onrender.com`. Confirm it with `curl https://<your-service>.onrender.com/healthz`.
4. Render's free plan spins the service down after 15 minutes idle; the first request after that takes ~30-50s to wake it back up. Fine for checking a forecast on your phone; not suited to always-on traffic.

**Frontend — Vercel (free static hosting)**

1. On [vercel.com](https://vercel.com), New Project → the same GitHub repo.
2. Set **Root Directory** to `dashboard` (Vercel auto-detects the Vite build otherwise; no `vercel.json` needed).
3. Add an environment variable `VITE_API_BASE_URL` = your Render URL from above (no trailing slash).
4. Deploy. Vercel gives you an `https://<project>.vercel.app` URL — open that on your phone.

Both providers' free tiers and exact steps change over time; if a screen doesn't match this description, follow the provider's own current instructions for "deploy a Vite/React static site" (Vercel) and "deploy a Python web service from a GitHub repo" (Render).

## Release gate: what must happen next

Do not promote the current model merely because its retrospective MAE is low. A defensible operational version needs all of the following:

1. Define one fixed forecast cutoff and local-day target contract, including source-publication latency.
2. Retrieve the matching NBM run directly from an immutable archive, recording source initialization time, valid time, retrieval time, model version, grid-cell policy, and checksum.
3. Log every prospective forecast before the target day, with a deterministic fallback to raw NBM if required source fields are unavailable.
4. Verify mature forecasts against NCEI `TMAX` across seasons and compare paired errors with raw NBM by station. Promote only after that predeclared evidence passes.

The supplied report's architecture is the reason this project uses a numerical-weather baseline plus a station-aware residual correction. See [ANALYSIS.md](ANALYSIS.md) for the full claim audit, evidence limits, and recommended hardening path.

## Key artifacts

| Artifact | Purpose |
|---|---|
| `artifacts/backtest/rolling_predictions.parquet` | Prediction-level, non-overlapping rolling-origin evidence. |
| `artifacts/backtest/overall_metrics.csv` | Aggregate comparison metrics. |
| `artifacts/backtest/station_metrics.csv` | Station-specific diagnostic metrics. |
| `artifacts/backtest/block_bootstrap_ci.csv` | Date-block bootstrap intervals. |
| `artifacts/backtest/acceptance.json` | Release decision and explicit blocker. |
| `artifacts/quality/TRAINING_DATA_QUALITY.md` | Input-table quality report. |
| `artifacts/production/xgb_residual_tmax.joblib` | Serialized shadow candidate. |
| `artifacts/production/model_manifest.json` | Training and release metadata. |
| `reports/MODEL_CARD.md` | Concise model card regenerated from current backtest artifacts. |

## External references

- [NOAA/NCEI Daily Summaries service](https://www.ncei.noaa.gov/access/services/data/v1)
- [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
- [NOAA National Blend of Models open-data registry](https://registry.opendata.aws/noaa-nbm/)
- [NOAA NBM download documentation](https://vlab.noaa.gov/web/mdl/nbm-download)

## Market-station extension (September 2026)

The settlement registry now has the 20 documented market locations and their
exact ICAO stations.  It is intentionally separate from model coverage: the
checked-in daily-high artifact remains calibrated only for KNYC, KMIA, KMDW,
KLAX, and KSFO.  A configured station without a validated calibration artifact
is rendered as `NO BET / INSUFFICIENT DATA`; it does not inherit a nearby
city's forecast.

| Data role | Source | Current use |
|---|---|---|
| Settlement-station registry | Local validated configuration | ICAO, coordinates, timezone, aliases, NWS grid, and settlement warning |
| Live observations | NWS station observations API | Intraday preliminary high/low/current temperature with timestamps |
| Final daily high label | NOAA/NCEI Daily Summaries | Existing model target and backtest truth |
| Forecast/grid data | NWS points/hourly/grid endpoints | Official-source client and provenance-ready integration |
| Numerical baseline | Open-Meteo public NBM/HRRR/GFS feeds | Existing model's live and archived input contract |

No API key is required for the current NWS, NCEI, or Open-Meteo integrations.
Set a descriptive contact in the `NWSClient` user agent before a production
deployment.  The app has no approved PrizePicks data adapter configured; use
the validated manual CSV/JSON import path in `polyweather.markets` or an
approved provider adapter. It does not scrape private endpoints or bypass
authentication.

Supported data/schema market types are daily high, daily low, temperature
bracket, threshold greater-than-or-equal, threshold less-than-or-equal, and an
explicitly rule-bound hourly temperature type.  Precipitation, snow, wind, and
severe-weather types are modeled as extensible enum values but are **not**
forecasted or settlement-verified by the shipped artifact.  Daily-low and
hourly numerical calibration likewise need dedicated historical artifacts
before the quality gate may return a strong prediction.
