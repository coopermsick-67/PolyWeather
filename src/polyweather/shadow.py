"""Immutable prospective forecast logging and later NCEI verification."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import MODEL_SOURCES, fetch_live_forecast_features, fetch_ncei_daily_tmax, has_complete_core_guidance
from .metrics import forecast_metrics, interval_metrics
from .model import ResidualForecaster
from .stations import Station


def create_shadow_records(
    model: ResidualForecaster,
    stations: Iterable[Station],
    target_date: date,
    *,
    model_artifact_sha256: str | None = None,
    feature_schema_version: str = "v003_20station_regime_features",
) -> list[dict]:
    """Create immutable, fully-provenanced forecasts for later scoring.

    A shadow record is valid only when the live feature contract matches the
    evaluated residual MOS (NBM + HRRR + GFS).  Missing guidance is not
    imputed into a supposedly prospective performance record.
    """
    issued_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    for station in stations:
        local_today = pd.Timestamp.now(tz=station.timezone).date()
        if target_date <= local_today:
            raise ValueError(
                f"{station.icao}: prospective logging requires a target date after local today "
                f"({local_today.isoformat()}); received {target_date.isoformat()}."
            )
        features = fetch_live_forecast_features(station, target_date)
        if not has_complete_core_guidance(features):
            raise ValueError(
                f"{station.icao}: refusing shadow snapshot without complete NBM, HRRR, and GFS guidance."
            )
        output = model.predict(pd.DataFrame([features])).iloc[0].to_dict()
        records.append(
            {
                "station": station.icao,
                "ghcn_id": station.ghcn_id,
                "target_date": target_date.isoformat(),
                "target_definition": "NCEI daily-summaries TMAX (official daily maximum)",
                "issue_time_utc": issued_at,
                "issue_time_contract": "latest live multi-model forecast at request time",
                "source_models": ["NCEP NBM CONUS", "NCEP HRRR CONUS", "NCEP GFS Seamless"],
                "required_source_models": list(MODEL_SOURCES),
                "guidance_complete": True,
                "record_schema_version": "v2_prospective_full_guidance",
                "model_family": f"{model.kind.title()} residual MOS",
                "model_identity": {
                    "artifact_sha256": model_artifact_sha256,
                    "kind": model.kind,
                    "train_rows": int(model.train_rows),
                    "calibration_rows": int(model.calibration_rows),
                    "feature_schema_version": feature_schema_version,
                },
                "nbm_baseline_f": float(features["nbm_baseline_f"]),
                "forecast_f": float(output["prediction_f"]),
                "p10_f": float(output["p10_f"]),
                "p50_f": float(output["p50_f"]),
                "p90_f": float(output["p90_f"]),
                "calibration_offset_f": float(output["calibration_offset_f"]),
                "conformal_halfwidth_f": float(output["conformal_halfwidth_f"]),
                "source_provenance": {
                    "provider": features.get("source_provider"),
                    "fetched_at_utc": features.get("source_fetched_at_utc"),
                    "generationtime_ms": features.get("source_generationtime_ms"),
                    "timezone": features.get("source_timezone"),
                    "utc_offset_seconds": features.get("source_utc_offset_seconds"),
                    "run_id": None,
                    "run_id_note": "The standard Open-Meteo endpoint does not expose a stable forecast-run ID.",
                },
                # The full predictor snapshot prevents a later upstream API
                # update from silently rewriting what was known at issue time.
                "feature_snapshot": {
                    key: value
                    for key, value in features.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
            }
        )
    return records


def append_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    """Append one auditable forecast per station/date/issue contract.

    A duplicate could silently turn a prospective record into cherry-picked
    reissues, so the operation rejects it rather than choosing a winner.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(records)
    existing_keys: set[tuple[str, str, str]] = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing_keys.add(
                    (str(record["station"]), str(record["target_date"]), str(record["issue_time_contract"]))
                )
    new_keys: set[tuple[str, str, str]] = set()
    for record in materialized:
        key = (str(record["station"]), str(record["target_date"]), str(record["issue_time_contract"]))
        if key in existing_keys or key in new_keys:
            raise ValueError(
                "Refusing duplicate prospective snapshot for "
                f"station={key[0]}, target_date={key[1]}, contract={key[2]!r}."
            )
        new_keys.add(key)
    with target.open("a", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return target


def verify_shadow_log(path: str | Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Join mature forecast snapshots to NCEI truth and compute audit metrics."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Shadow log not found: {source}")
    records: list[dict] = []
    malformed_lines: list[int] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            malformed_lines.append(line_number)
    if malformed_lines:
        logging.getLogger(__name__).error(
            "Skipped %d malformed line(s) in shadow log %s: %s", len(malformed_lines), source, malformed_lines
        )
    if not records:
        raise ValueError("Shadow log contains no forecast records.")
    forecasts = pd.DataFrame(records)
    forecasts["target_date"] = pd.to_datetime(forecasts["target_date"]).dt.date
    mature = forecasts.loc[forecasts["target_date"] < date.today()].copy()
    if mature.empty:
        return mature, {"n": 0}
    # The caller's records include official IDs, so reconstruct lightweight
    # station objects only through package registry at the module boundary.
    from .stations import STATIONS

    requested = [STATIONS[station] for station in sorted(mature["station"].unique())]
    labels = fetch_ncei_daily_tmax(requested, mature["target_date"].min(), mature["target_date"].max())
    verified = mature.merge(labels[["station", "target_date", "tmax_f"]], on=["station", "target_date"], how="inner")
    metrics = forecast_metrics(verified, "tmax_f", "forecast_f")
    metrics.update(interval_metrics(verified, actual="tmax_f", lower="p10_f", upper="p90_f"))
    return verified, metrics
