"""High-signal data-quality checks for the model-training table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .model import BASELINE_COLUMN, TARGET_COLUMN


def assess_training_table(table: pd.DataFrame) -> dict[str, object]:
    """Check grain, coverage, missingness, plausibility, and feature availability."""
    data = table.copy()
    data["target_date"] = pd.to_datetime(data["target_date"])
    required = ["station", "target_date", TARGET_COLUMN, BASELINE_COLUMN]
    missing_required = {column: int(data[column].isna().sum()) for column in required if column in data}
    numeric = data.select_dtypes(include="number")
    missing_rates = (numeric.isna().mean().sort_values(ascending=False) * 100).round(2)
    low_availability = int((data.get("ncep_nbm_conus__availability", pd.Series(1.0, index=data.index)) < 0.90).sum())
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
        "nbm_profile_availability_below_90pct_rows": low_availability,
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
        f"- NBM profiles below 90% hourly availability: {report['nbm_profile_availability_below_90pct_rows']}",
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
