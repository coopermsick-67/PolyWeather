"""High-signal data-quality checks for the model-training table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MODEL_SOURCES
from .model import BASELINE_COLUMN, SUPPORTED_FORECAST_LEAD_DAYS, TARGET_COLUMN


def assess_training_table(table: pd.DataFrame) -> dict[str, object]:
    """Check grain, coverage, missingness, plausibility, and feature availability."""
    data = table.copy()
    data["target_date"] = pd.to_datetime(data["target_date"])
    required = ["station", "target_date", TARGET_COLUMN, BASELINE_COLUMN]
    absent_required = [column for column in required if column not in data]
    if absent_required:
        raise ValueError(f"Training table is missing required columns: {', '.join(absent_required)}")
    missing_required = {column: int(data[column].isna().sum()) for column in required if column in data}
    numeric = data.select_dtypes(include="number")
    missing_rates = (numeric.isna().mean().sort_values(ascending=False) * 100).round(2)
    availability_rows = {
        model: int((pd.to_numeric(data.get(f"{model}__availability", pd.Series(np.nan, index=data.index)), errors="coerce") < 0.90).sum())
        for model in MODEL_SOURCES
    }
    core_columns = [f"{model}__tmax_f" for model in MODEL_SOURCES]
    missing_core_guidance = int(
        (~np.isfinite(data.reindex(columns=core_columns).apply(pd.to_numeric, errors="coerce"))).any(axis=1).sum()
    )
    all_missing_numeric = sorted(column for column in numeric if numeric[column].isna().all())
    leads = sorted(pd.to_numeric(data.get("forecast_lead_days", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    unsupported_leads = [lead for lead in leads if lead != SUPPORTED_FORECAST_LEAD_DAYS]
    daily_station_counts = data.groupby("target_date")["station"].nunique()
    station_gap_rows = []
    for station, group in data.groupby("station", sort=True):
        dates = pd.DatetimeIndex(group["target_date"].dropna().sort_values().unique())
        expected_days = (dates.max() - dates.min()).days + 1 if len(dates) else 0
        station_gap_rows.append({"station": str(station), "missing_calendar_days": int(expected_days - len(dates))})
    station_rows = (
        data.groupby("station")
        .agg(
            rows=("target_date", "size"),
            first_date=("target_date", "min"),
            last_date=("target_date", "max"),
            missing_tmax=(TARGET_COLUMN, lambda series: int(series.isna().sum())),
            missing_nbm=(BASELINE_COLUMN, lambda series: int(series.isna().sum())),
        )
        .reset_index()
    )
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "grain": "one station x local target date x fixed 24-hour archived forecast lead",
        "date_min": str(data["target_date"].min().date()),
        "date_max": str(data["target_date"].max().date()),
        "duplicate_station_date_rows": int(data.duplicated(["station", "target_date"]).sum()),
        "missing_required": missing_required,
        "tmax_range_f": [float(data[TARGET_COLUMN].min()), float(data[TARGET_COLUMN].max())],
        "nbm_baseline_range_f": [float(data[BASELINE_COLUMN].min()), float(data[BASELINE_COLUMN].max())],
        "implausible_tmax_rows": int(((data[TARGET_COLUMN] < -60) | (data[TARGET_COLUMN] > 140)).sum()),
        "implausible_nbm_rows": int(((data[BASELINE_COLUMN] < -80) | (data[BASELINE_COLUMN] > 140)).sum()),
        "profile_availability_below_90pct_rows": availability_rows,
        "rows_missing_any_core_model_tmax": missing_core_guidance,
        "all_missing_numeric_columns": all_missing_numeric,
        "forecast_lead_days": leads,
        "unsupported_forecast_lead_days": unsupported_leads,
        "station_count_per_day_min": int(daily_station_counts.min()),
        "station_count_per_day_max": int(daily_station_counts.max()),
        "station_calendar_gaps": station_gap_rows,
        "training_table_hash_values": int(data.get("training_table_sha256", pd.Series(dtype=str)).replace("", pd.NA).nunique(dropna=True)),
        "station_coverage": station_rows.assign(
            first_date=station_rows["first_date"].dt.date.astype(str),
            last_date=station_rows["last_date"].dt.date.astype(str),
        ).to_dict(orient="records"),
        "top_feature_missing_rates_pct": missing_rates.head(15).to_dict(),
    }


def write_quality_report(table: pd.DataFrame, output_dir: str | Path) -> Path:
    """Persist machine-readable and reviewer-friendly quality findings."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = assess_training_table(table)
    json_path = directory / "training_data_quality.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Training-data quality report",
        "",
        f"- Grain: {report['grain']}",
        f"- Coverage: {report['rows']:,} rows, {report['date_min']} to {report['date_max']}",
        f"- Duplicate station/date rows: {report['duplicate_station_date_rows']}",
        f"- TMAX range: {report['tmax_range_f'][0]:.1f} to {report['tmax_range_f'][1]:.1f} °F",
        f"- NBM baseline range: {report['nbm_baseline_range_f'][0]:.1f} to {report['nbm_baseline_range_f'][1]:.1f} °F",
        f"- Implausible labels/baselines: {report['implausible_tmax_rows']} / {report['implausible_nbm_rows']}",
        f"- Rows missing any core-model Tmax: {report['rows_missing_any_core_model_tmax']}",
        f"- Forecast lead days present: {report['forecast_lead_days']} (unsupported: {report['unsupported_forecast_lead_days']})",
        f"- All-missing numeric columns excluded by training: {len(report['all_missing_numeric_columns'])}",
        f"- Profiles below 90% hourly availability: {report['profile_availability_below_90pct_rows']}",
        "",
        "## Station coverage",
        "",
        "| station | rows | first date | last date | missing Tmax | missing NBM |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in report["station_coverage"]:
        lines.append(
            f"| {row['station']} | {row['rows']} | {row['first_date']} | {row['last_date']} | "
            f"{row['missing_tmax']} | {row['missing_nbm']} |"
        )
    lines.extend(["", "## Caveat", "", "A valid row means the public archived forecast fields and NCEI daily label joined successfully. It does not prove a single frozen daily forecast issuance; that is tracked separately in shadow operation."])
    directory.joinpath("TRAINING_DATA_QUALITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path
