"""Build the bounded Data Analytics artifact for the PolyWeather technical report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKTEST = ROOT / "artifacts" / "backtest"
QUALITY = ROOT / "artifacts" / "quality" / "training_data_quality.json"
OUTPUT = ROOT / "reports" / "technical_report_artifact.json"


def _rows(frame: pd.DataFrame) -> list[dict]:
    """Return JSON-safe records while keeping reviewed values at full precision."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def main() -> None:
    overall = pd.read_csv(BACKTEST / "overall_metrics.csv")
    station = pd.read_csv(BACKTEST / "station_metrics.csv")
    acceptance = json.loads((BACKTEST / "acceptance.json").read_text(encoding="utf-8"))
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))

    xgb = overall.loc[overall["model"] == "XGBoost residual"].iloc[0]
    nbm = overall.loc[overall["model"] == "NBM"].iloc[0]
    station_models = station.loc[station["model"].isin(["NBM", "XGBoost residual"])].copy()
    station_models["model_display"] = station_models["model"].replace(
        {"NBM": "Raw NBM baseline", "XGBoost residual": "XGBoost residual"}
    )
    station_models["mae_reduction_vs_nbm_f"] = station_models.groupby("station")["mae_f"].transform(
        lambda values: float(values.max() - values.min())
    )
    wide = station_models.pivot(index="station", columns="model", values=["mae_f", "bias_f", "n", "coverage"])
    station_audit = pd.DataFrame(
        {
            "station": wide.index,
            "nbm_mae_f": wide[("mae_f", "NBM")].to_numpy(),
            "xgb_mae_f": wide[("mae_f", "XGBoost residual")].to_numpy(),
            "xgb_bias_f": wide[("bias_f", "XGBoost residual")].to_numpy(),
            "xgb_coverage": wide[("coverage", "XGBoost residual")].to_numpy(),
            "n": wide[("n", "XGBoost residual")].to_numpy(),
        }
    )
    station_audit["mae_skill_vs_nbm"] = 1 - station_audit["xgb_mae_f"] / station_audit["nbm_mae_f"]
    station_audit["mae_reduction_f"] = station_audit["nbm_mae_f"] - station_audit["xgb_mae_f"]
    station_audit = station_audit.sort_values("mae_skill_vs_nbm", ascending=False).reset_index(drop=True)

    backtest_overall_source = {
        "id": "backtest_overall",
        "label": "PolyWeather corrected rolling-origin overall backtest",
        "path": "artifacts/backtest/rolling_predictions.parquet",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "description": "Calendar-ordered, expanding-origin forecast evaluation using contiguous 31-day folds. Each station/local target date appears once in the held-out set.",
            "sql": """WITH p AS (\n  SELECT * FROM read_parquet('artifacts/backtest/rolling_predictions.parquet')\n)\nSELECT\n  COUNT(*) AS heldout_station_dates,\n  COUNT(DISTINCT target_date) AS heldout_target_dates,\n  AVG(ABS(tmax_f - nbm_baseline_f)) AS nbm_mae_f,\n  AVG(ABS(tmax_f - xgb_prediction_f)) AS xgb_mae_f,\n  1 - AVG(ABS(tmax_f - xgb_prediction_f)) / AVG(ABS(tmax_f - nbm_baseline_f)) AS mae_skill_vs_nbm,\n  AVG(CASE WHEN tmax_f BETWEEN p10_f AND p90_f THEN 1.0 ELSE 0.0 END) AS interval_coverage\nFROM p;""",
            "tables_used": [
                "artifacts/backtest/rolling_predictions.parquet",
                "artifacts/backtest/overall_metrics.csv",
                "artifacts/backtest/station_metrics.csv",
                "artifacts/backtest/block_bootstrap_ci.csv",
                "artifacts/backtest/acceptance.json",
            ],
            "filters": [
                "Five stations: KNYC, KMIA, KMDW, KLAX, KSFO",
                "Held-out target dates 2025-04-06 through 2026-08-10",
                "One 24-hour archived forecast lead per local target-date/hour",
            ],
            "metric_definitions": [
                "MAE (degrees F) = mean(abs(official NCEI TMAX - forecast)).",
                "MAE skill versus NBM = 1 - MAE_model / MAE_NBM.",
                "Coverage = fraction of NCEI TMAX outcomes inside the nominal 80% XGBoost split-conformal interval.",
                "Bootstrap intervals resample entire target dates, retaining all station rows within each sampled date.",
            ],
        },
    }
    backtest_station_source = {
        "id": "backtest_station",
        "label": "PolyWeather corrected rolling-origin station backtest",
        "path": "artifacts/backtest/rolling_predictions.parquet",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "description": "Station-level error, bias, skill, interval coverage, and sample-size calculation over the same unique held-out forecast rows.",
            "sql": """WITH p AS (\n  SELECT * FROM read_parquet('artifacts/backtest/rolling_predictions.parquet')\n)\nSELECT\n  station,\n  COUNT(*) AS n,\n  AVG(ABS(tmax_f - nbm_baseline_f)) AS nbm_mae_f,\n  AVG(ABS(tmax_f - xgb_prediction_f)) AS xgb_mae_f,\n  AVG(xgb_prediction_f - tmax_f) AS xgb_bias_f,\n  1 - AVG(ABS(tmax_f - xgb_prediction_f)) / AVG(ABS(tmax_f - nbm_baseline_f)) AS mae_skill_vs_nbm,\n  AVG(CASE WHEN tmax_f BETWEEN p10_f AND p90_f THEN 1.0 ELSE 0.0 END) AS xgb_coverage\nFROM p\nGROUP BY station\nORDER BY mae_skill_vs_nbm DESC;""",
            "tables_used": ["artifacts/backtest/rolling_predictions.parquet"],
            "filters": [
                "Five stations: KNYC, KMIA, KMDW, KLAX, KSFO",
                "Held-out target dates 2025-04-06 through 2026-08-10",
                "One 24-hour archived forecast lead per local target-date/hour",
            ],
            "metric_definitions": [
                "Station MAE (degrees F) = mean(abs(official NCEI TMAX - forecast)) within each station.",
                "Station skill = 1 - station MAE_XGBoost / station MAE_NBM.",
                "Station coverage = fraction of official TMAX outcomes inside the nominal 80% XGBoost interval.",
            ],
        },
    }
    quality_source = {
        "id": "training_quality",
        "label": "PolyWeather training-table quality checks",
        "path": "artifacts/quality/training_data_quality.json",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "description": "Completeness, duplication, plausibility, and profile-availability checks for the joined training table.",
            "sql": """WITH t AS (\n  SELECT * FROM read_parquet('data/features/tmax_24h_composite_training.parquet')\n)\nSELECT\n  COUNT(*) AS rows,\n  MIN(target_date) AS date_min,\n  MAX(target_date) AS date_max,\n  COUNT(*) - COUNT(DISTINCT (station, target_date)) AS duplicate_station_date_rows,\n  SUM(CASE WHEN tmax_f IS NULL THEN 1 ELSE 0 END) AS missing_tmax_rows,\n  SUM(CASE WHEN nbm_baseline_f IS NULL THEN 1 ELSE 0 END) AS missing_nbm_rows\nFROM t;""",
            "tables_used": [
                "data/features/tmax_24h_composite_training.parquet",
                "artifacts/quality/training_data_quality.json",
            ],
            "filters": ["Rows with an official NCEI daily Tmax label and archived forecast join only"],
            "metric_definitions": [
                "Training grain is one station × local target date × fixed 24-hour archived forecast lead.",
                "Duplicate check is computed on station plus local target date.",
            ],
        },
    }
    report_source = {
        "id": "architecture_report",
        "label": "User-supplied daily-temperature system report",
        "path": "C:/Users/cool7/Downloads/Building a Daily Temperature Prediction System for KNYC, KMIA, KMDW, KLAX, and KSFO.pdf",
        "query": {
            "description": "Architecture proposal audited to set target definition, residual-MOS design, rolling validation, provenance, and release gates. It is not used as empirical evidence for the backtest score.",
        },
    }
    sources = [backtest_overall_source, backtest_station_source, quality_source, report_source]

    snapshot = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "datasets": {
            "headline": [
                {
                    "xgb_mae_f": float(xgb["mae_f"]),
                    "nbm_mae_f": float(nbm["mae_f"]),
                    "mae_skill_vs_nbm": float(xgb["mae_skill_vs_nbm"]),
                    "minimum_skill_gate": float(acceptance["minimum_global_mae_skill_vs_nbm"]),
                    "interval_coverage": float(xgb["coverage"]),
                    "nominal_coverage": 0.80,
                    "heldout_station_dates": int(xgb["n"]),
                    "heldout_target_dates": 492,
                    "release_status": acceptance["decision"],
                }
            ],
            "station_model_mae": _rows(station_models),
            "station_audit": _rows(station_audit),
            "model_comparison": _rows(overall),
            "training_quality": [
                {
                    "rows": quality["rows"],
                    "columns": quality["columns"],
                    "date_min": quality["date_min"],
                    "date_max": quality["date_max"],
                    "duplicate_station_date_rows": quality["duplicate_station_date_rows"],
                    "implausible_tmax_rows": quality["implausible_tmax_rows"],
                    "implausible_nbm_rows": quality["implausible_nbm_rows"],
                    "low_nbm_availability_rows": quality["nbm_profile_availability_below_90pct_rows"],
                }
            ],
        },
    }

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "PolyWeather Daily High Temperature Forecast: Technical Evaluation",
        "description": "Corrected rolling-origin assessment of a five-station daily maximum temperature residual-MOS candidate.",
        "generatedAt": snapshot["generatedAt"],
        "sources": sources,
        "cards": [
            {
                "id": "xgb_mae",
                "description": "Mean absolute error against the official local-day NCEI Tmax label across unique held-out station/date forecasts.",
                "dataset": "headline",
                "sourceId": "backtest_overall",
                "metrics": [
                    {"label": "XGBoost MAE (degrees F)", "field": "xgb_mae_f", "format": "number"},
                    {"label": "Raw NBM MAE (degrees F)", "field": "nbm_mae_f", "format": "number"},
                ],
            },
            {
                "id": "skill",
                "description": "Relative reduction in mean absolute error compared with the unmodified NBM hourly-profile maximum.",
                "dataset": "headline",
                "sourceId": "backtest_overall",
                "metrics": [
                    {"label": "MAE skill vs. NBM", "field": "mae_skill_vs_nbm", "format": "percent"},
                    {"label": "Minimum skill gate", "field": "minimum_skill_gate", "format": "percent"},
                ],
            },
            {
                "id": "coverage",
                "description": "Observed coverage of the nominal 80% split-conformal prediction interval in the rolling held-out evaluation.",
                "dataset": "headline",
                "sourceId": "backtest_overall",
                "metrics": [
                    {"label": "Interval coverage", "field": "interval_coverage", "format": "percent"},
                    {"label": "Nominal coverage", "field": "nominal_coverage", "format": "percent"},
                ],
            },
            {
                "id": "sample",
                "description": "Unique station/local-date held-out forecasts in the corrected rolling evaluation.",
                "dataset": "headline",
                "sourceId": "backtest_overall",
                "metrics": [
                    {"label": "Held-out station-dates", "field": "heldout_station_dates", "format": "number"},
                    {"label": "Held-out target dates", "field": "heldout_target_dates", "format": "number"},
                ],
            },
        ],
        "charts": [
            {
                "id": "station_mae_comparison",
                "title": "Held-out daily Tmax MAE by station and forecast",
                "subtitle": "492 local target dates per station (491 for KMIA); values are degrees F and lower is better.",
                "type": "bar",
                "intent": "comparison",
                "question": "How does the residual candidate compare with the raw NBM baseline at each station?",
                "rationale": "Grouped bars make the within-station absolute-error comparison clear while keeping the shared zero baseline honest.",
                "dataset": "station_model_mae",
                "sourceId": "backtest_station",
                "layout": "full",
                "palette": {"kind": "categorical"},
                "settings": {"groupMode": "grouped", "orientation": "vertical", "sort": "none", "showValues": True},
                "encodings": {
                    "x": {"field": "station", "type": "nominal", "label": "Station"},
                    "y": {"field": "mae_f", "type": "quantitative", "format": "number", "label": "MAE", "unit": "degrees F"},
                    "color": {"field": "model_display", "type": "nominal"},
                    "tooltip": [
                        {"field": "station", "type": "nominal"},
                        {"field": "model_display", "type": "nominal"},
                        {"field": "mae_f", "type": "quantitative", "format": "number", "unit": "degrees F"},
                        {"field": "mae_skill_vs_nbm", "type": "quantitative", "format": "percent"},
                        {"field": "n", "type": "quantitative", "format": "number"},
                    ],
                },
            }
        ],
        "tables": [
            {
                "id": "station_audit_table",
                "title": "Station-level audit detail",
                "subtitle": "Unique held-out station/local-date forecasts; positive skill and reduction favor XGBoost.",
                "dataset": "station_audit",
                "sourceId": "backtest_station",
                "layout": "full",
                "density": "spacious",
                "defaultSort": {"field": "mae_skill_vs_nbm", "direction": "desc"},
                "columns": [
                    {"field": "station", "label": "Station", "type": "text"},
                    {"field": "nbm_mae_f", "label": "Raw NBM MAE (degrees F)", "type": "number", "format": "number"},
                    {"field": "xgb_mae_f", "label": "XGBoost MAE (degrees F)", "type": "number", "format": "number"},
                    {"field": "mae_reduction_f", "label": "MAE reduction (degrees F)", "type": "number", "format": "number", "movement": True},
                    {"field": "mae_skill_vs_nbm", "label": "MAE skill vs. NBM", "type": "percent", "format": "percent", "movement": True},
                    {"field": "xgb_bias_f", "label": "XGBoost bias (degrees F)", "type": "number", "format": "number", "movement": True},
                    {"field": "xgb_coverage", "label": "80% interval coverage", "type": "percent", "format": "percent"},
                    {"field": "n", "label": "Held-out n", "type": "number", "format": "number"},
                ],
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# PolyWeather Daily High Temperature Forecast: Technical Evaluation"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": "## Technical summary\n\n**The candidate materially improves the historical NBM baseline, but it is not yet a production forecast system.** On 2,459 unique held-out station/local-date forecasts from 6 April 2025 through 10 August 2026, XGBoost residual MOS achieved 1.82 degrees F MAE versus 2.65 degrees F for the raw NBM profile maximum: 31.2% MAE skill. It improves on NBM at all five stations.\n\nThe release decision is **SHADOW_ONLY**. The retrospective features are leakage-resistant fixed 24-hour-lead hourly values, but they are not a reconstruction of one frozen, scheduled daily issuance. Prospective fixed-cutoff logging and later NCEI verification are required before any production accuracy claim."},
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["xgb_mae", "skill", "coverage", "sample"]},
            {
                "id": "station_finding",
                "type": "markdown",
                "body": "## Gains are real at every station, but they are not equally strong\n\n**KLAX and KMIA account for the largest relative improvement; KMDW, KNYC, and KSFO have narrower margins.** The chart compares raw NBM and the residual candidate within each station, so it should be read as a local calibration result rather than a universal difficulty ranking. XGBoost wins globally by only 0.05 degrees F versus Ridge, and Ridge is better at KLAX and KSFO; station-level monitoring therefore matters more than the global average alone."},
            {"id": "station_chart", "type": "chart", "chartId": "station_mae_comparison"},
            {
                "id": "station_table_context",
                "type": "markdown",
                "body": "## Station-level evidence supports shadow monitoring, not a uniform release\n\n**Every station clears the raw-NBM comparison, while uncertainty and residual bias vary.** The audit table keeps the underlying error, skill, bias, coverage, and sample-size evidence together. The nominal 80% bands covered 86.5% overall, which is conservative in this historical slice rather than proof of future calibration."},
            {"id": "station_table", "type": "table", "tableId": "station_audit_table"},
            {
                "id": "scope_and_definitions",
                "type": "markdown",
                "sourceId": "training_quality",
                "body": "## Scope and measurement contract\n\n**The target is an official daily label, not a reconstructed hourly maximum.** The training table joins NOAA/NCEI Daily Summaries TMAX for each station's local calendar date to archived NCEP forecast profiles. It contains 3,359 complete station-date rows from 8 October 2024 through 10 August 2026, with zero duplicate station/date rows and no missing required TMAX or NBM baseline values.\n\nMAE is the primary metric: mean absolute difference, in degrees Fahrenheit, between predicted and official daily Tmax. Skill is `1 - MAE_model / MAE_NBM`; positive values are better than the unmodified NBM hourly-profile maximum. The model's 80% interval is a split-conformal residual band calibrated only on earlier dates."},
            {
                "id": "model_and_validation",
                "type": "markdown",
                "body": "## The build follows the report's residual-MOS and anti-leakage logic\n\n**The deployed candidate predicts the daily-Tmax residual rather than replacing numerical weather prediction.** It starts from the archived NBM hourly 2 m temperature-profile maximum, then learns a station-aware XGBoost correction using NBM, HRRR, and GFS temperature-curve, humidity, cloud, wind, precipitation, pressure/radiation availability, and seasonal features. A Ridge residual MOS is retained as the linear challenger.\n\nThe evaluation is expanding-origin and strictly calendar ordered: each fold trains only on earlier target dates, holds a trailing earlier calibration block out of model fitting, and tests on the next contiguous 31 days. An earlier implementation used month starts with fixed 31-day windows; it was corrected before handoff because short months could overlap the next fold. The delivered result contains zero duplicated held-out station/date forecasts."},
            {
                "id": "limits_and_uncertainty",
                "type": "markdown",
                "body": "## The score is credible for the retrospective contract, not a promise of future accuracy\n\n**The decisive limitation is forecast-vintage identity.** The archived data use one 24-hour lead per valid hour and aggregate those hourly values into the local-day high. That avoids reanalysis leakage, but it is still an hour-wise composite rather than one real-world daily forecast run frozen at a declared issue time.\n\nThe original report correctly calls for immutable issue/valid/ingest provenance, separate NWS versus NBM baselines, and station/regime slices. The present prototype does not yet include direct run-level NBM GRIB neighborhood extraction, a historical immutable NWS forecastGridData archive, ASOS state, radar, satellite, SST, or long prospective monitoring. The bootstrap 95% MAE interval is 1.76 to 1.89 degrees F and skill interval is 29.0% to 33.2%, but neither resolves those operational gaps."},
            {
                "id": "recommended_next_steps",
                "type": "markdown",
                "body": "## Recommended next steps\n\n1. Freeze one local issue schedule per horizon and write every forecast, input snapshot, model hash, and target definition before the outcome occurs.\n2. Replace the retrospective composite with direct NOAA NBM GRIB run extraction; retain valid time, cycle, forecast hour, ingest time, and the local station-neighborhood features.\n3. Score those immutable logs against the same NCEI TMAX label through all seasonal transitions, then compare separately with raw NBM and an independently frozen NWS forecast-grid baseline.\n4. Promote only if the fixed-cutoff candidate continues to beat both raw NBM and Ridge in rolling and prospective station-level results with calibrated intervals."
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": "## Further questions\n\n- Does a fixed morning issuance retain the same skill, especially after removing any same-day trajectory information?\n- Can station-specific blending outperform one global XGBoost candidate, given Ridge's advantage at KLAX and KSFO?\n- Which local-regime features most improve the weak-margin sites: marine layer at KLAX/KSFO, lake breeze or snow at KMDW, and coastal-convective regimes at KMIA?\n- How stable is calibration after upstream NBM version and source-model changes?"
            },
        ],
    }
    artifact = {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}
    OUTPUT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "datasets": {name: len(rows) for name, rows in snapshot["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
