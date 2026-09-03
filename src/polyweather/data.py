"""Source clients and cutoff-safe model-training table construction.

The historical NCEP values are retrieved from Open-Meteo's public Previous
Model Runs endpoint.  Each ``*_previous_dayN`` field is an archived forecast
at a fixed lead, so it is materially different from a reanalysis field that
already knows the outcome.  The target remains NOAA/NCEI daily-summary TMAX.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .stations import STATIONS, Station, require_station, station_metadata as registry_station_metadata
from .nws import NWSClient, daily_observation_extremes, is_stale, normalized_observations


NCEI_DAILY_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NWS_OBSERVATIONS_URL = "https://api.weather.gov/stations/{station}/observations"

# The report calls for NBM as baseline and HRRR/GFS as complementary inputs.
# The provider's canonical model identifiers are intentionally centralized.
MODEL_SOURCES = ("ncep_nbm_conus", "ncep_hrrr_conus", "ncep_gfs_seamless")
ARCHIVED_HOURLY_VARIABLES = (
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "shortwave_radiation",
    "surface_pressure",
)
LIVE_HOURLY_VARIABLES = ARCHIVED_HOURLY_VARIABLES


class SourceError(RuntimeError):
    """Raised when an upstream weather source cannot provide required data."""


def has_complete_core_guidance(features: dict[str, object]) -> bool:
    """Return whether every model required by the residual MOS supplied Tmax.

    The fitted residual model was evaluated with NBM, HRRR, and GFS together.
    Scikit-learn can impute a missing source, but doing that live would silently
    change the deployed forecast contract.  Callers must therefore use the
    residual correction only when this small, non-imputed core is present.
    """
    for model in MODEL_SOURCES:
        value = features.get(f"{model}__tmax_f")
        try:
            if not np.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


class _RequestPacer:
    """Process-wide pacing for public weather endpoints.

    Previous Runs is a shared public service. A multi-station backfill must
    pace *all* worker threads together or it will be throttled and produce a
    selectively incomplete historical sample.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._next_allowed_at = 0.0

    def wait(self, minimum_interval_seconds: float) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_allowed_at - now)
            self._next_allowed_at = max(now, self._next_allowed_at) + minimum_interval_seconds
        if delay:
            time.sleep(delay)


_PREVIOUS_RUNS_PACER = _RequestPacer()


def _request_json(
    url: str,
    params: dict,
    retries: int = 6,
    minimum_interval_seconds: float = 0.0,
) -> dict | list:
    error: Exception | None = None
    headers = {"User-Agent": "PolyWeather/0.1 (research temperature forecast system)"}
    for attempt in range(retries):
        try:
            if minimum_interval_seconds:
                _PREVIOUS_RUNS_PACER.wait(minimum_interval_seconds)
            response = requests.get(url, params=params, headers=headers, timeout=90)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    pause = max(float(retry_after), 1.0) if retry_after else min(60.0, 4.0 * (attempt + 1))
                except ValueError:
                    pause = min(60.0, 4.0 * (attempt + 1))
                if attempt + 1 < retries:
                    time.sleep(pause)
                    continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise SourceError(payload.get("reason", "Unknown weather API error"))
            return payload
        except (requests.RequestException, ValueError, SourceError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(45.0, 1.5 * (attempt + 1)))
    raise SourceError(f"Request failed for {url}: {error}") from error


def _date_range(start: date, end: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def fetch_ncei_daily_tmax(stations: Iterable[Station], start: date, end: date) -> pd.DataFrame:
    """Download official NCEI GHCN-Daily Tmax labels in Fahrenheit.

    NCEI daily-summaries with ``units=standard`` supplies Fahrenheit values.
    Missing or malformed daily highs are deliberately discarded rather than
    synthesized from forecast/reanalysis data.
    """
    station_list = list(stations)
    params = {
        "dataset": "daily-summaries",
        "stations": ",".join(station.ghcn_id for station in station_list),
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "format": "json",
        "units": "standard",
        "includeAttributes": "true",
    }
    records = _request_json(NCEI_DAILY_URL, params)
    if not isinstance(records, list):
        raise SourceError("NCEI daily-summaries returned an unexpected payload.")
    by_ghcn = {station.ghcn_id: station.icao for station in station_list}
    rows: list[dict] = []
    for record in records:
        station_id = record.get("STATION")
        high = pd.to_numeric(record.get("TMAX"), errors="coerce")
        if station_id not in by_ghcn or pd.isna(high):
            continue
        rows.append(
            {
                "station": by_ghcn[station_id],
                "ghcn_id": station_id,
                "target_date": pd.Timestamp(record["DATE"]).date(),
                "tmax_f": float(high),
                "tmax_attributes": record.get("TMAX_ATTRIBUTES", ""),
                "target_definition": "NCEI daily-summaries TMAX (official daily maximum)",
            }
        )
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise SourceError("NCEI returned no usable TMAX labels for the requested range.")
    return labels.sort_values(["station", "target_date"]).reset_index(drop=True)


def _archived_variable_name(variable: str, lead_days: int, model: str, multi_model: bool) -> str:
    suffix = f"_previous_day{lead_days}"
    return f"{variable}{suffix}_{model}" if multi_model else f"{variable}{suffix}"


def _summarize_hourly_forecast(
    payload: dict,
    station: Station,
    lead_days: int,
    models: tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate archived hourly forecast curves to station-local daily features."""
    hourly = payload.get("hourly") or {}
    local_time = pd.to_datetime(hourly.get("time", []), errors="coerce")
    if len(local_time) == 0:
        return pd.DataFrame()
    frame = pd.DataFrame({"local_time": local_time})
    frame["target_date"] = frame["local_time"].dt.date
    for model in models:
        for variable in ARCHIVED_HOURLY_VARIABLES:
            column = _archived_variable_name(variable, lead_days, model, len(models) > 1)
            values = hourly.get(column)
            if values is None:
                # Provider may not have a requested variable/model in older windows.
                frame[f"{model}__{variable}"] = np.nan
            else:
                frame[f"{model}__{variable}"] = pd.to_numeric(values, errors="coerce")

    result_rows: list[dict] = []
    for target_date, day in frame.groupby("target_date", sort=True):
        row: dict[str, object] = {
            "station": station.icao,
            "target_date": target_date,
            "timezone": station.timezone,
            "forecast_lead_days": lead_days,
            "forecast_lead_hours": lead_days * 24,
        }
        for model in models:
            temperature = day[f"{model}__temperature_2m"]
            row[f"{model}__tmax_f"] = float(temperature.max()) if temperature.notna().any() else np.nan
            row[f"{model}__tmin_f"] = float(temperature.min()) if temperature.notna().any() else np.nan
            row[f"{model}__temp_mean_f"] = float(temperature.mean()) if temperature.notna().any() else np.nan
            # Preserve a compact diurnal curve. Hour slots make the boosted model
            # sensitive to late cloud clearing / marine-layer timing.
            for hour in (0, 3, 6, 9, 12, 15, 18, 21):
                values = day.loc[day["local_time"].dt.hour == hour, f"{model}__temperature_2m"]
                row[f"{model}__temp_{hour:02d}_f"] = float(values.iloc[0]) if values.notna().any() else np.nan
            for variable in ARCHIVED_HOURLY_VARIABLES:
                values = day[f"{model}__{variable}"]
                prefix = f"{model}__{variable}"
                if variable == "wind_direction_10m":
                    radians = np.deg2rad(values)
                    row[f"{prefix}_sin_mean"] = float(np.nanmean(np.sin(radians))) if values.notna().any() else np.nan
                    row[f"{prefix}_cos_mean"] = float(np.nanmean(np.cos(radians))) if values.notna().any() else np.nan
                elif variable == "precipitation":
                    row[f"{prefix}_sum"] = float(values.sum(min_count=1))
                    row[f"{prefix}_max"] = float(values.max()) if values.notna().any() else np.nan
                else:
                    row[f"{prefix}_mean"] = float(values.mean()) if values.notna().any() else np.nan
                    row[f"{prefix}_max"] = float(values.max()) if values.notna().any() else np.nan
                    row[f"{prefix}_min"] = float(values.min()) if values.notna().any() else np.nan
            row[f"{model}__availability"] = float(temperature.notna().mean())
        result_rows.append(row)
    return pd.DataFrame(result_rows)


def fetch_archived_forecast_features(
    stations: Iterable[Station],
    start: date,
    end: date,
    lead_days: int = 1,
    models: tuple[str, ...] = MODEL_SOURCES,
    chunk_days: int = 31,
    max_workers: int = 2,
    request_interval_seconds: float = 0.8,
    cache_dir: str | Path | None = "data/raw/open_meteo_previous_runs",
) -> pd.DataFrame:
    """Retrieve vintage-safe station-local forecast features in bounded chunks.

    ``previous_day1`` is a 24-hour-ahead archived value for each target hour;
    the API's fixed-lead alignment makes the predictor unavailable future
    outcome information. For this first production-ready prototype, all
    stations are evaluated at the same locked 24-hour lead.
    """
    station_list = list(stations)
    if not station_list:
        raise ValueError("At least one settlement station is required.")
    all_frames: list[pd.DataFrame] = []
    model_parameter = ",".join(models)
    variables = ",".join(f"{variable}_previous_day{lead_days}" for variable in ARCHIVED_HOURLY_VARIABLES)

    cache_root = Path(cache_dir) if cache_dir else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)

    def fetch_one(chunk_start: date, chunk_end: date) -> pd.DataFrame:
        # Open-Meteo accepts coordinate lists and responds with one payload per
        # location. Batching the configured settlement stations cuts a
        # 20-city rebuild from hundreds of public API calls to one call per
        # date chunk, which is both more reliable and friendlier to the source.
        params = {
            "latitude": ",".join(str(station.latitude) for station in station_list),
            "longitude": ",".join(str(station.longitude) for station in station_list),
            "hourly": variables,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
            "models": model_parameter,
        }
        cache_key = hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        cache_path = cache_root / f"batch_{chunk_start}_{chunk_end}_{cache_key[:12]}.json" if cache_root else None
        if cache_path and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = _request_json(
                OPEN_METEO_PREVIOUS_RUNS_URL,
                params,
                minimum_interval_seconds=request_interval_seconds,
            )
            if cache_path:
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
                os.replace(temporary, cache_path)
        payloads = payload if isinstance(payload, list) else [payload]
        if len(payloads) != len(station_list) or not all(isinstance(item, dict) for item in payloads):
            raise SourceError("Previous-runs API returned an unexpected multi-location payload.")
        frames = [
            _summarize_hourly_forecast(location_payload, station, lead_days, models)
            for station, location_payload in zip(station_list, payloads, strict=True)
        ]
        return pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()

    jobs = list(_date_range(start, end, chunk_days))
    # Bounded concurrency makes the multi-year data build practical while
    # respecting a public endpoint. Results remain deterministically sorted
    # below, regardless of response order.
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one, *job) for job in jobs]
        for future in as_completed(futures):
            summarized = future.result()
            if not summarized.empty:
                all_frames.append(summarized)
            completed += 1
            print(f"Archived forecast chunks: {completed}/{len(jobs)}", flush=True)
    if not all_frames:
        raise SourceError("Historical forecast source returned no usable forecast features.")
    result = pd.concat(all_frames, ignore_index=True)
    return result.sort_values(["station", "target_date"]).drop_duplicates(["station", "target_date"])


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    day_of_year = pd.to_datetime(result["target_date"]).dt.dayofyear.astype(float)
    result["dayofyear_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["dayofyear_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    result["month"] = pd.to_datetime(result["target_date"]).dt.month.astype(int)
    return result


def add_derived_forecast_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic forecast-agreement and diurnal-shape features.

    Every source field is already available at issuance.  These compact
    differences make inter-model disagreement and the expected daytime
    warming profile explicit for the residual learner, without adding any
    observation, analysis, or future-outcome data.
    """
    result = frame.copy()
    # The raw hourly summaries are intentionally retained; these additions
    # expose physically meaningful relationships that a shallow residual
    # learner otherwise has to rediscover from sparse station-level history.
    # Every term below is calculated only from the archived/live forecast
    # fields available at issuance time.
    for model in MODEL_SOURCES:
        prefix = f"{model}__"
        tmax = f"{prefix}tmax_f"
        tmin = f"{prefix}tmin_f"
        tmean = f"{prefix}temp_mean_f"
        dewpoint = f"{prefix}dew_point_2m_mean"
        cloud = f"{prefix}cloud_cover_mean"
        humidity = f"{prefix}relative_humidity_2m_mean"
        wind_speed = f"{prefix}wind_speed_10m_mean"
        wind_sin = f"{prefix}wind_direction_10m_sin_mean"
        wind_cos = f"{prefix}wind_direction_10m_cos_mean"
        if {tmax, tmin}.issubset(result):
            result[f"{prefix}diurnal_range_f"] = result[tmax] - result[tmin]
        if {tmax, tmean}.issubset(result):
            result[f"{prefix}peak_above_mean_f"] = result[tmax] - result[tmean]
        if {tmean, dewpoint}.issubset(result):
            result[f"{prefix}dewpoint_depression_f"] = result[tmean] - result[dewpoint]
        if {cloud, humidity}.issubset(result):
            result[f"{prefix}cloud_humidity"] = result[cloud] * result[humidity] / 100
        if {wind_speed, wind_sin}.issubset(result):
            result[f"{prefix}wind_u_mean"] = result[wind_speed] * result[wind_sin]
        if {wind_speed, wind_cos}.issubset(result):
            result[f"{prefix}wind_v_mean"] = result[wind_speed] * result[wind_cos]
    for suffix in ("tmax_f", "tmin_f", "temp_mean_f"):
        columns = [f"{model}__{suffix}" for model in MODEL_SOURCES if f"{model}__{suffix}" in result]
        if len(columns) > 1:
            result[f"model_agreement__{suffix}_spread"] = result[columns].max(axis=1) - result[columns].min(axis=1)
            result[f"model_agreement__{suffix}_std"] = result[columns].std(axis=1)
        baseline = f"ncep_nbm_conus__{suffix}"
        for model in MODEL_SOURCES[1:]:
            other = f"{model}__{suffix}"
            if baseline in result and other in result:
                short_name = model.removeprefix("ncep_").removesuffix("_conus").removesuffix("_seamless")
                result[f"model_agreement__nbm_minus_{short_name}__{suffix}"] = result[baseline] - result[other]
        if suffix == "tmax_f" and len(columns) == len(MODEL_SOURCES):
            nbm, hrrr, gfs = columns
            result["model_agreement__weighted_consensus_tmax_f"] = .5 * result[nbm] + .25 * result[hrrr] + .25 * result[gfs]
            result["model_agreement__hrrr_minus_gfs__tmax_f"] = result[hrrr] - result[gfs]
            result["model_agreement__nbm_position_tmax_f"] = result[nbm] - result[columns].mean(axis=1)
    for model in MODEL_SOURCES:
        early = [f"{model}__temp_{hour:02d}_f" for hour in (0, 3, 6, 9) if f"{model}__temp_{hour:02d}_f" in result]
        late = [f"{model}__temp_{hour:02d}_f" for hour in (12, 15, 18, 21) if f"{model}__temp_{hour:02d}_f" in result]
        if early and late:
            morning = result[early].mean(axis=1)
            afternoon = result[late].mean(axis=1)
            result[f"{model}__morning_mean_f"] = morning
            result[f"{model}__afternoon_mean_f"] = afternoon
            result[f"{model}__warming_f"] = afternoon - morning
    result["feature_schema_version"] = "v003_20station_regime_features"
    return result


def build_training_table(
    start: date,
    end: date,
    lead_days: int = 1,
    stations: Iterable[Station] | None = None,
    chunk_days: int = 31,
    max_workers: int = 2,
    request_interval_seconds: float = 0.8,
    cache_dir: str | Path | None = "data/raw/open_meteo_previous_runs",
) -> pd.DataFrame:
    """Build a joined, audit-friendly TMAX table with no label fabrication."""
    station_list = list(stations or STATIONS.values())
    labels = fetch_ncei_daily_tmax(station_list, start, end)
    forecasts = fetch_archived_forecast_features(
        station_list,
        start,
        end,
        lead_days=lead_days,
        chunk_days=chunk_days,
        max_workers=max_workers,
        request_interval_seconds=request_interval_seconds,
        cache_dir=cache_dir,
    )
    table = labels.merge(forecasts, on=["station", "target_date"], how="inner", validate="one_to_one")
    table = add_derived_forecast_features(_add_calendar_features(table))
    # The archived NBM hourly profile is the unmodified physical/NWP baseline.
    table["nbm_baseline_f"] = table["ncep_nbm_conus__tmax_f"]
    table["issue_time_contract"] = f"fixed {lead_days * 24}h archived lead"
    table["feature_schema_version"] = "v003_20station_regime_features"
    table["label_schema_version"] = "v001_ncei_daily_summaries_tmax"
    table["source_vintage"] = "Open-Meteo Previous Runs fixed-lead archive"
    table["training_table_sha256"] = ""
    return table.sort_values(["target_date", "station"]).reset_index(drop=True)


def write_training_table(table: pd.DataFrame, path: str | Path) -> Path:
    """Persist Parquet plus a deterministic integrity hash in the table."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    prepared = table.copy()
    hash_frame = prepared.drop(columns=["training_table_sha256"], errors="ignore")
    digest = hashlib.sha256(pd.util.hash_pandas_object(hash_frame, index=True).values.tobytes()).hexdigest()
    prepared["training_table_sha256"] = digest
    prepared.to_parquet(target, index=False)
    metadata = {
        "path": str(target),
        "rows": int(len(prepared)),
        "sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stations": sorted(prepared["station"].unique().tolist()),
        "date_min": str(prepared["target_date"].min()),
        "date_max": str(prepared["target_date"].max()),
        "source_vintage": "Open-Meteo Previous Runs fixed-lead archive",
        "label_source": "NOAA/NCEI daily-summaries",
    }
    target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def fetch_live_forecast_features(station: Station, target_date: date) -> dict[str, object]:
    """Fetch the current NBM/HRRR/GFS forecast curve for one local target date.

    Live source data are only for inference. They are never mixed into the
    retrospective table, which avoids a hidden look-ahead path.
    """
    params = {
        "latitude": station.latitude,
        "longitude": station.longitude,
        "hourly": ",".join(LIVE_HOURLY_VARIABLES),
        "temperature_unit": "fahrenheit",
        "timezone": station.timezone,
        "forecast_days": 16,
        "models": ",".join(MODEL_SOURCES),
    }
    payload = _request_json(OPEN_METEO_FORECAST_URL, params)
    if not isinstance(payload, dict):
        raise SourceError("Live forecast API returned an unexpected payload.")
    # Live multi-model responses use ``variable_model`` naming. Convert to the
    # exact shape expected by the archived-feature summarizer by temporarily
    # renaming the requested day's columns as a single archived lead field.
    hourly = payload.get("hourly") or {}
    transformed: dict[str, list] = {"time": hourly.get("time", [])}
    for model in MODEL_SOURCES:
        for variable in LIVE_HOURLY_VARIABLES:
            source_column = f"{variable}_{model}"
            transformed[_archived_variable_name(variable, 1, model, True)] = hourly.get(source_column, [])
    synthetic = {"hourly": transformed}
    summarized = _summarize_hourly_forecast(synthetic, station, 1, MODEL_SOURCES)
    selected = summarized.loc[summarized["target_date"] == target_date]
    if selected.empty:
        raise SourceError(f"Live forecast does not cover {target_date.isoformat()} for {station.icao}.")
    row = selected.iloc[0].to_dict()
    row = add_derived_forecast_features(_add_calendar_features(pd.DataFrame([row]))).iloc[0].to_dict()
    row["nbm_baseline_f"] = row.get("ncep_nbm_conus__tmax_f", np.nan)
    local_today = datetime.now(ZoneInfo(station.timezone)).date()
    row["forecast_lead_days"] = max((target_date - local_today).days, 0)
    row["forecast_lead_hours"] = row["forecast_lead_days"] * 24
    row["issue_time_contract"] = "current latest forecast; generated at request time"
    row["feature_schema_version"] = "v003_20station_regime_features"
    # Open-Meteo's standard endpoint does not expose a stable individual-run
    # identifier. Preserve the facts it *does* provide so an immutable shadow
    # record can distinguish source retrieval time from forecast target time.
    row["source_provider"] = "Open-Meteo Forecast API"
    row["source_fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    row["source_generationtime_ms"] = payload.get("generationtime_ms")
    row["source_timezone"] = str(payload.get("timezone") or station.timezone)
    row["source_utc_offset_seconds"] = payload.get("utc_offset_seconds")
    row["guidance_complete"] = has_complete_core_guidance(row)
    return row


def fetch_nws_observed_high_f(station: Station, target_date: date) -> float | None:
    """Return the observed high so far for a station-local day, if available.

    This is used only as a lower bound when predicting *today*. It prevents a
    morning/afternoon forecast from being lower than a temperature already
    measured at the target station; it is not used as a retrospective feature.
    """
    local_start = datetime.combine(target_date, datetime.min.time(), tzinfo=station.tzinfo)
    local_end = local_start + timedelta(days=1)
    params = {
        "start": local_start.astimezone(timezone.utc).isoformat(),
        "end": min(datetime.now(timezone.utc), local_end.astimezone(timezone.utc)).isoformat(),
        "limit": 500,
    }
    try:
        payload = _request_json(NWS_OBSERVATIONS_URL.format(station=station.icao), params, retries=1)
    except SourceError:
        return None
    if not isinstance(payload, dict):
        return None
    temperatures = []
    for feature in payload.get("features", []):
        value = ((feature.get("properties") or {}).get("temperature") or {}).get("value")
        if value is not None:
            temperatures.append(float(value) * 9 / 5 + 32)
    return max(temperatures) if temperatures else None


def fetch_live_station_snapshot(station: Station, target_date: date, client: NWSClient | None = None) -> dict[str, object]:
    """Fetch official NWS live inputs with source timing and stale-data flags.

    These observations are preliminary intraday evidence only.  Final daily
    settlement must use the configured climate/NCEI source from market rules.
    """
    nws = client or NWSClient()
    local_start = datetime.combine(target_date, datetime.min.time(), tzinfo=station.tzinfo)
    local_end = local_start + timedelta(days=1)
    observation_snapshot = nws.observations(station, local_start, min(local_end, datetime.now(station.tzinfo)))
    rows = normalized_observations(observation_snapshot, station)
    high, low = daily_observation_extremes(rows, station, target_date)
    latest = rows[-1] if rows else None
    return {
        "stationId": station.icao,
        "timezone": station.timezone,
        "observedDailyHighF": high,
        "observedDailyLowF": low,
        "currentObservedTemperatureF": latest["temperature_f"] if latest else None,
        "lastSuccessfulObservationTime": latest["observed_at"].isoformat() if latest else None,
        "stale": is_stale(observation_snapshot),
        "sourceTimestamp": observation_snapshot.source_time.isoformat() if observation_snapshot.source_time else None,
        "source": "NWS station observations",
        "preliminary": True,
        "observations": [
            {"time": item["local_time"].isoformat(), "temperatureF": round(float(item["temperature_f"]), 1)}
            for item in rows[-96:]
        ],
    }


def station_metadata() -> list[dict]:
    """Return serializable station metadata for manifests and reports."""
    return registry_station_metadata()
