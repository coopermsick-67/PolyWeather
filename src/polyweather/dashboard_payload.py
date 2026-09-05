"""Small JSON adapter used by the local Vite dashboard development server."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from . import bet_evidence
from .betfilter import (
    BetEvidence,
    DataQuality,
    ForecastSnapshot,
    analyze_ensemble,
    analyze_observation,
    analyze_stability,
    decide,
    from_calibrated_interval,
    from_residual_history,
    summarize,
)
from .betfilter.config import DEFAULT_CONFIG, BetFilterConfig
from .data import MODEL_SOURCES, fetch_live_forecast_features, fetch_live_station_snapshot, has_complete_core_guidance
from .markets import QualityStatus, quality_gate
from .model import (
    MIN_PREDICTION_FEATURE_COMPLETENESS,
    SUPPORTED_FORECAST_LEAD_DAYS,
    AdaptiveResidualForecaster,
    BlendedResidualForecaster,
    ResidualForecaster,
    feature_completeness,
)
from .stations import STATIONS, station_metadata


ROOT = Path(__file__).resolve().parents[2]
# The local/Render adapter uses the same audited v4 evidence family throughout.
MODEL_PATH = ROOT / "artifacts" / "production_v4" / "xgb_residual_tmax.joblib"
PREDICTIONS_PATH = ROOT / "artifacts" / "backtest_v4" / "rolling_predictions.parquet"
STATION_METRICS_PATH = ROOT / "artifacts" / "backtest_v4" / "station_metrics.csv"
OVERALL_METRICS_PATH = ROOT / "artifacts" / "backtest_v4" / "overall_metrics.csv"
DASHBOARD_TIMEZONE = ZoneInfo("America/New_York")
STABILITY_STATE_PATH = ROOT / "data" / "normalized" / "dashboard_forecast_state.json"
MIN_STABLE_CHANGE_F = 2
FOUR_DEGREE_HALF_WIDTH_F = 2
STABILITY_STATE_RETENTION_DAYS = 45


def _date(value: str) -> date:
    return date.fromisoformat(value)


def dashboard_today() -> date:
    return datetime.now(DASHBOARD_TIMEZONE).date()


def _earliest_station_local_today() -> date:
    """The earliest "today" across every configured station's own timezone.

    A West Coast station's local calendar day has not yet rolled over for
    up to ~3 hours after America/New_York already has (e.g. 9:00-11:59 PM
    Pacific is already after midnight Eastern). Anchoring date validation
    to America/New_York alone would reject a still-current Pacific "today"
    during that window. Every configured station is within the continental
    US, so no station's local today can be more than one day behind or
    ahead of any other's.
    """
    now = datetime.now(timezone.utc)
    return min(now.astimezone(station.tzinfo).date() for station in STATIONS.values())


def validate_dashboard_date(target_date: date) -> None:
    earliest_today = _earliest_station_local_today()
    latest_today = dashboard_today()
    if not earliest_today <= target_date <= latest_today + timedelta(days=7):
        raise ValueError(
            f"Choose a date from {earliest_today.isoformat()} through {(latest_today + timedelta(days=7)).isoformat()}."
        )


STALE_LOCK_AGE_S = 30.0


@contextlib.contextmanager
def _state_file_lock(path: Path, timeout_s: float = 5.0) -> Iterator[None]:
    """A simple cross-platform exclusive lock so concurrent workers do not
    interleave read-modify-write cycles on the shared stability-state file.

    A lock older than STALE_LOCK_AGE_S is assumed to be an orphan left by a
    crashed/killed worker (this critical section does no network I/O, so a
    legitimately held lock is never anywhere near that old) and is reclaimed
    rather than left to stall every future request for timeout_s each time.
    """
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    fd = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            try:
                age_s = time.time() - lock_path.stat().st_mtime
            except OSError:
                age_s = 0.0
            if age_s > STALE_LOCK_AGE_S:
                with contextlib.suppress(OSError):
                    lock_path.unlink()
                continue
            if time.monotonic() >= deadline:
                # Proceed without the lock rather than blocking a request
                # forever on a stale lock file.
                break
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
            with contextlib.suppress(OSError):
                lock_path.unlink()


def _load_stability_state(path: Path = STABILITY_STATE_PATH) -> dict[str, Any]:
    """Read the local display-state cache without making it a failure point."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) and isinstance(payload.get("forecasts"), dict) else {"forecasts": {}}
    except (OSError, ValueError, json.JSONDecodeError):
        return {"forecasts": {}}


def _write_stability_state(state: dict[str, Any], path: Path = STABILITY_STATE_PATH) -> None:
    """Atomically persist display continuity across refreshes and restarts."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # The forecast remains usable if a local cache cannot be updated.
        return


def _prune_stability_state(state: dict[str, Any], today: date) -> None:
    """Bound the local continuity cache instead of retaining every forecast forever."""
    cutoff = today - timedelta(days=STABILITY_STATE_RETENTION_DAYS)
    forecasts = state.get("forecasts", {})
    for key in list(forecasts):
        # Current keys are model-hash::target-date::station. Older dashboard
        # versions used target-date::station, so support both while pruning.
        parts = str(key).split("::")
        date_part = parts[-2] if len(parts) >= 3 else parts[0]
        try:
            if date.fromisoformat(date_part) < cutoff:
                forecasts.pop(key, None)
        except ValueError:
            # A malformed cache entry cannot safely provide display continuity.
            forecasts.pop(key, None)


def stabilize_display_high(
    previous: int | None,
    candidate: int,
    observed_high: int | None = None,
    minimum_change_f: int = MIN_STABLE_CHANGE_F,
) -> tuple[int, str]:
    """Avoid noisy one-degree refresh churn while allowing meaningful changes.

    A high already observed today always wins. Otherwise, display updates need
    a two-degree change from the last displayed value; the underlying model is
    still recomputed on every refresh and recorded in the state cache.
    """
    if observed_high is not None:
        candidate = max(candidate, observed_high)
    if previous is None:
        return candidate, "initial"
    if observed_high is not None and observed_high > previous:
        return candidate, "observed_high"
    if abs(candidate - previous) < minimum_change_f:
        return previous, "held_minor_change"
    return candidate, "material_model_change"


# These three read fixed backtest-artifact files (PREDICTIONS_PATH,
# STATION_METRICS_PATH, OVERALL_METRICS_PATH) that only change when a new
# model is deployed, which restarts the process and clears this cache.
# Without caching, every single /api/dashboard request re-read and
# re-parsed the same CSV/parquet files for no reason.
@functools.lru_cache(maxsize=1)
def _trend() -> dict[str, list[float] | list[str]]:
    rows = pd.read_parquet(PREDICTIONS_PATH).copy()
    rows["target_date"] = pd.to_datetime(rows["target_date"])
    rows["candidate_error"] = (rows["xgb_prediction_f"] - rows["tmax_f"]).abs()
    rows["baseline_error"] = (rows["nbm_baseline_f"] - rows["tmax_f"]).abs()
    daily = rows.groupby("target_date")[["candidate_error", "baseline_error"]].mean().rolling(14, min_periods=7).mean().dropna().tail(14)
    return {
        "labels": [f"{timestamp.strftime('%b')} {timestamp.day}" for timestamp in daily.index],
        "candidate": [round(float(value), 2) for value in daily["candidate_error"]],
        "baseline": [round(float(value), 2) for value in daily["baseline_error"]],
    }


@functools.lru_cache(maxsize=1)
def _accuracy() -> list[dict[str, Any]]:
    """Expose the audited station ranking rather than presenting a generic score."""
    metrics = pd.read_csv(STATION_METRICS_PATH)
    candidate = metrics.loc[metrics["model"].eq("XGBoost residual")].copy()
    candidate = candidate.sort_values("mae_f", kind="stable").reset_index(drop=True)
    return [
        {
            "rank": int(index + 1),
            "station": str(row.station),
            "maeF": round(float(row.mae_f), 2),
            "p90ErrorF": round(float(row.p90_ae_f), 2),
            "within2Pct": int(round(float(row.within_2f) * 100)),
            "biasF": round(float(row.bias_f), 2),
        }
        for index, row in candidate.iterrows()
    ]


@functools.lru_cache(maxsize=1)
def _model_evidence() -> dict[str, float | int]:
    metrics = pd.read_csv(OVERALL_METRICS_PATH)
    candidate = metrics.loc[metrics["model"].eq("XGBoost residual")].iloc[0]
    baseline = metrics.loc[metrics["model"].eq("NBM")].iloc[0]
    return {
        "candidateMaeF": round(float(candidate.mae_f), 2),
        "baselineMaeF": round(float(baseline.mae_f), 2),
        "skillPct": int(round(float(candidate.mae_skill_vs_nbm) * 100)),
        "within2Pct": int(round(float(candidate.within_2f) * 100)),
        "testForecasts": int(candidate["n"]),
        # A fixed ±2°F range is the requested four-degree planning band. Its
        # coverage is measured directly on untouched XGBoost forecasts rather
        # than inferred from the model's variable-width conformal interval.
        "fourDegreeCoveragePct": int(round(float(candidate.within_2f) * 100)),
        "calibratedCoveragePct": int(round(float(candidate.coverage) * 100)),
        "calibratedMeanWidthF": round(float(candidate.mean_width_f), 1),
    }


def _fetch_live_inputs(station: Any, target_date: date) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """Fetch one station's independent live inputs concurrently with the others."""
    features = fetch_live_forecast_features(station, target_date)
    snapshot = fetch_live_station_snapshot(station, target_date) if target_date == datetime.now(station.tzinfo).date() else None
    return station, features, snapshot


def _model_stations(model_path: Path) -> list[Any]:
    """Use the artifact's trained station set, not every configured station.

    A missing or malformed manifest means we cannot verify which stations the
    loaded model was actually trained on, so it is safer to treat *no*
    station as calibrated than to silently apply an unvalidated correction
    to every configured station.
    """
    manifest_path = model_path.with_name("model_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        codes = [str(code).upper() for code in manifest["stations"]]
        selected = [STATIONS[code] for code in codes if code in STATIONS]
        if selected:
            return selected
        raise ValueError("model_manifest.json listed no recognized stations")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.getLogger(__name__).error(
            "Could not read trained-station list from %s (%s); disabling model calibration for this request.",
            manifest_path, exc,
        )
        return []


def _select_model_path() -> Path:
    """Return the declared v3 artifact and fail closed when it is absent.

    Serving a plausible number from an old model with different station
    coverage is more dangerous than surfacing an operational error.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Required v4 20-station model artifact is missing: {MODEL_PATH}. "
            "Rebuild production_v4 before serving forecasts."
        )
    return MODEL_PATH


def payload(target_date: date) -> dict:
    validate_dashboard_date(target_date)
    model_path = _select_model_path()
    # joblib.load runs pickle deserialization, which can execute arbitrary
    # code for a crafted file. Safe only because model_path is fixed to
    # artifacts/ committed by this repo's own training pipeline, never a
    # user- or request-supplied path. Do not let model_path become dynamic
    # without adding integrity verification (e.g. a signed checksum).
    model: ResidualForecaster | AdaptiveResidualForecaster | BlendedResidualForecaster = joblib.load(model_path)
    model_version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    model_stations = _model_stations(model_path)
    trained_station_ids = {station.icao for station in model_stations}
    missing_trained_stations = sorted(set(STATIONS) - trained_station_ids)
    if missing_trained_stations:
        logging.getLogger(__name__).error(
            "Production model manifest does not cover all configured stations; missing: %s",
            ", ".join(missing_trained_stations),
        )
    # The UI needs every configured market.  Untrained stations deliberately
    # receive a live numerical baseline and a NO BET status rather than being
    # hidden or assigned an unvalidated residual-model correction.
    display_stations = list(STATIONS.values())
    # Network retrieval dominates response time. Each station uses a separate
    # endpoint request, so running these in parallel removes the avoidable
    # five-station waterfall without changing the forecast contract. This is
    # done outside the state-file lock so concurrent requests don't serialize
    # on network I/O, only on the brief local read-modify-write below.
    with ThreadPoolExecutor(max_workers=min(8, len(display_stations))) as executor:
        future_by_station = {
            executor.submit(_fetch_live_inputs, station, target_date): station for station in display_stations
        }
        live_inputs = []
        for future in future_by_station:
            try:
                live_inputs.append(future.result())
            except Exception:  # noqa: BLE001 - one station's upstream failure must not sink the whole payload
                station = future_by_station[future]
                logging.getLogger(__name__).exception("Live data fetch failed for %s; omitting from this response.", station.icao)
    with _state_file_lock(STABILITY_STATE_PATH):
        stability_state = _load_stability_state()
        _prune_stability_state(stability_state, dashboard_today())
        saved_forecasts: dict[str, Any] = stability_state["forecasts"]
        forecasts, stability_state = _build_forecasts(
            live_inputs, model, trained_station_ids, model_version, target_date, saved_forecasts, stability_state
        )
        _write_stability_state(stability_state)
    return _assemble_payload(target_date, forecasts, model_version)


SNAPSHOT_HISTORY_LIMIT = 24
BET_FILTER_MODE = os.environ.get("WEATHERPICKS_BET_FILTER_MODE", "conservative")
BET_FILTER_ENABLED = os.environ.get("WEATHERPICKS_BET_FILTER", "1") != "0"
# Single source of truth for whether a provider-specific market settlement
# contract (exact station, product, rounding rule, and entry cutoff) has been
# verified for these forecasts. Every station maps to a documented NOAA/NCEI
# settlement ICAO, but that is a data-quality fact, not a confirmed betting
# contract. Before this was split into two independent, disagreeing values --
# `quality_gate` was told `rules_available=False` (forcing the *display*
# status to NO BET) while `DataQuality.settlement_station_verified` was
# hardcoded `True` (so the *decision engine's* own hard gate never enforced
# this at all). Flip this only after a specific provider's contract has
# actually been confirmed.
SETTLEMENT_CONTRACT_VERIFIED = False


def _bet_filter_config() -> BetFilterConfig:
    try:
        return BetFilterConfig().for_mode(BET_FILTER_MODE)  # type: ignore[arg-type]
    except ValueError:
        logging.getLogger(__name__).warning(
            "Unknown bet-filter mode %r; falling back to conservative.", BET_FILTER_MODE
        )
        return DEFAULT_CONFIG


def _record_snapshot(
    state_entry: dict[str, Any], predicted_high_f: float, bucket_lower_f: int | None
) -> list[ForecastSnapshot]:
    """Append this refresh to the station-day's forecast trail and return it.

    Stability cannot be measured from a single value, so the trail is the
    only thing that makes "has this forecast stopped moving?" answerable.
    History is appended to, never rewritten: overwriting past snapshots
    would erase exactly the revisions the filter needs to see.
    """
    history = list(state_entry.get("snapshots", []))
    history.append({
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "predictedHighF": round(float(predicted_high_f), 2),
        "bucketLowerF": bucket_lower_f,
    })
    state_entry["snapshots"] = history[-SNAPSHOT_HISTORY_LIMIT:]
    parsed: list[ForecastSnapshot] = []
    for item in state_entry["snapshots"]:
        try:
            parsed.append(ForecastSnapshot(
                captured_at=datetime.fromisoformat(str(item["capturedAt"])),
                predicted_high_f=float(item["predictedHighF"]),
                bucket_lower_f=item.get("bucketLowerF"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


def _bet_decision(
    station: Any,
    features: dict[str, Any],
    observation: dict[str, Any] | None,
    predicted_high_f: float,
    interval_lower_f: float | None,
    interval_upper_f: float | None,
    is_calibrated: bool,
    supported_horizon: bool,
    completeness: float,
    target_date: date,
    is_same_day: bool,
    snapshots: list[ForecastSnapshot],
    config: BetFilterConfig,
) -> dict[str, Any] | None:
    """Run the decision layer for one station and return its verdict.

    Returns ``None`` only when the filter is switched off. Every other path
    -- including missing inputs -- produces a real decision, because
    "we could not evaluate this" is itself an answer the board must show
    rather than an excuse to fall back to a bare forecast.
    """
    if not BET_FILTER_ENABLED:
        return None
    icao = station.icao
    observed_high = observation.get("observedDailyHighF") if observation else None
    if is_calibrated and interval_lower_f is not None and interval_upper_f is not None:
        residuals = bet_evidence.residuals_for(icao)
        try:
            if residuals is not None:
                distribution = from_residual_history(
                    predicted_high_f, residuals,
                    observed_high_floor_f=observed_high if is_same_day else None,
                )
            else:
                raise ValueError("no residual history")
        except ValueError:
            distribution = from_calibrated_interval(
                predicted_high_f, interval_lower_f, interval_upper_f, 0.80,
                observed_high_floor_f=observed_high if is_same_day else None,
            )
    else:
        # Without a validated interval there is no honest distribution to
        # build. A nominal one is created only so the gates can run and
        # report DATA_INSUFFICIENT with a real reason attached.
        distribution = from_calibrated_interval(predicted_high_f, predicted_high_f - 4, predicted_high_f + 4, 0.80)
    market_bucket = bet_evidence.bucket_for(predicted_high_f)
    ensemble = analyze_ensemble({
        source: features.get(f"{source}__tmax_f") for source in MODEL_SOURCES
    })
    stability = analyze_stability(snapshots)
    reliability = bet_evidence.reliability_for(icao)
    alignment = analyze_observation(
        is_same_day=is_same_day,
        predicted_high_f=predicted_high_f,
        current_temperature_f=observation.get("currentObservedTemperatureF") if observation else None,
        observed_high_so_far_f=observed_high,
        bucket_upper_f=market_bucket[1] + 0.5,
    )
    fetched_at = features.get("source_run_initialized_at_utc")
    age_minutes: float | None = None
    if fetched_at:
        try:
            age_minutes = (
                datetime.now(timezone.utc) - datetime.fromisoformat(str(fetched_at))
            ).total_seconds() / 60.0
        except (TypeError, ValueError):
            age_minutes = None
    horizon_hours = max(
        0.0, (datetime.combine(target_date, datetime.min.time(), tzinfo=station.tzinfo)
              + timedelta(hours=17) - datetime.now(station.tzinfo)).total_seconds() / 3600.0
    )
    evidence = BetEvidence(
        station=icao,
        target_date=target_date.isoformat(),
        is_same_day=is_same_day,
        distribution=distribution,
        ensemble=ensemble,
        stability=stability,
        reliability=reliability,
        observation=alignment,
        data_quality=DataQuality(
            is_calibrated=is_calibrated,
            supported_horizon=supported_horizon,
            feature_completeness=completeness,
            # Count distinct reported values, not raw source names: two
            # "independent" guidance sources that came back byte-identical
            # (see AUDIT.md item 1 -- HRRR/GFS matched exactly on 99.8% of
            # held-out rows) are one source duplicated, not two sources
            # corroborating each other.
            source_count=ensemble.distinct_value_count,
            # Every configured station maps to a documented settlement ICAO;
            # the market *contract* for it is a separate thing, and this must
            # bind the decision engine's own hard gate, not just the display
            # status computed below via `quality_gate`.
            settlement_station_verified=SETTLEMENT_CONTRACT_VERIFIED,
            data_age_minutes=age_minutes,
            forecast_horizon_hours=horizon_hours,
            source_run_age_verified=features.get("source_run_age_verified") is True,
            # The existing binary calibration artifact has no independent
            # later validation for the deployed empirical distribution.
            probability_calibration_verified=False,
        ),
        market_bucket=market_bucket,
        calibrator=bet_evidence.probability_calibrator(),
    )
    return decide(evidence, config).to_dict()


def _build_forecasts(
    live_inputs: list[tuple[Any, dict[str, Any], dict[str, Any] | None]],
    model: Any,
    trained_station_ids: set[str],
    model_version: str,
    target_date: date,
    saved_forecasts: dict[str, Any],
    stability_state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    forecasts: list[dict[str, Any]] = []
    filter_config = _bet_filter_config()
    for station, features, observation in live_inputs:
        icao = station.icao
        baseline_high = float(features.get("nbm_baseline_f", np.nan))
        complete_guidance = has_complete_core_guidance(features)
        lead_days = int(features.get("forecast_lead_days", -1))
        supported_horizon = lead_days == SUPPORTED_FORECAST_LEAD_DAYS
        completeness = float(feature_completeness(model, pd.DataFrame([features]))[0])
        features_sufficient = completeness >= MIN_PREDICTION_FEATURE_COMPLETENESS
        # The deployed residual correction was evaluated with all three
        # guidance sources. Do not let the estimator's missing-value imputer
        # turn a partial upstream response into a supposedly calibrated pick.
        is_calibrated = (
            icao in trained_station_ids
            and complete_guidance
            and supported_horizon
            and features_sufficient
        )
        prediction = model.predict(pd.DataFrame([features])).iloc[0] if is_calibrated else None
        predicted_high = float(prediction["prediction_f"]) if prediction is not None else baseline_high
        if not np.isfinite(predicted_high) or not np.isfinite(baseline_high):
            # Skip only this station rather than failing the whole dashboard
            # payload; the other stations' forecasts remain independently valid.
            logging.getLogger(__name__).error("Forecast inputs are incomplete for %s; omitting from this response.", icao)
            continue
        raw_model_high_rounded = int(np.rint(predicted_high))
        baseline_high_rounded = int(np.rint(baseline_high))
        observed_high = observation.get("observedDailyHighF") if observation else None
        observed_low = observation.get("observedDailyLowF") if observation else None
        current_temperature = observation.get("currentObservedTemperatureF") if observation else None
        candidate_high = int(np.rint(max(predicted_high, observed_high if observed_high is not None else float("-inf"))))
        observed_high_rounded = int(np.rint(observed_high)) if observed_high is not None else None
        # A model upgrade should not inherit a display value held from the
        # previous artifact. Within one artifact, the existing two-degree
        # stability rule still suppresses inconsequential refresh churn.
        state_key = f"{model_version}::{target_date.isoformat()}::{icao}"
        entry = dict(saved_forecasts.get(state_key, {}))
        previous = entry.get("display_high_f")
        high, stability_reason = stabilize_display_high(previous, candidate_high, observed_high_rounded)
        snapshots = _record_snapshot(entry, predicted_high, bet_evidence.bucket_for(predicted_high)[0])
        entry.update({
            "display_high_f": high,
            "candidate_high_f": candidate_high,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        saved_forecasts[state_key] = entry
        interval_lower = float(prediction["interval_lower_f"]) if prediction is not None else None
        interval_upper = float(prediction["interval_upper_f"]) if prediction is not None else None
        interval_width = interval_upper - interval_lower if prediction is not None else None
        half_width = interval_width / 2.0 if interval_width is not None else None
        disagreement_f = float(features.get("model_agreement__tmax_f_spread", np.nan))
        source_agreement = max(0.0, min(1.0, 1.0 - disagreement_f / 8.0)) if np.isfinite(disagreement_f) else 0.0
        freshness = 0.0 if observation and observation.get("stale") else 1.0
        # A station registry is not a market-contract registry. Until the
        # exact provider contract, settlement source, bucket rule, and target
        # date have been reviewed, weather output may be displayed but it is
        # never an actionable market pick.
        status = quality_gate(
            station_known=True, rules_available=SETTLEMENT_CONTRACT_VERIFIED, freshness=freshness,
            agreement=source_agreement, interval_width_f=interval_width,
        )
        if status is QualityStatus.UNKNOWN_SETTLEMENT_RULE or not is_calibrated or not complete_guidance:
            status = QualityStatus.NO_BET
        forecasts.append(
            {
                "station": icao,
                "city": station.name,
                "marketLocation": station.display_name,
                "targetDate": target_date.isoformat(),
                "settlementNote": station.display_note,
                "timezone": station.timezone,
                "marketType": "daily_high",
                "isCalibrated": is_calibrated,
                "forecastLeadDays": lead_days,
                "evaluatedLeadDays": SUPPORTED_FORECAST_LEAD_DAYS,
                "supportedHorizon": supported_horizon,
                "featureCompletenessPct": round(100 * completeness, 1),
                "guidanceComplete": complete_guidance,
                "requiredGuidanceModels": list(MODEL_SOURCES),
                "sourceProvenance": {
                    "provider": features.get("source_provider"),
                    "fetchedAt": features.get("source_fetched_at_utc"),
                    "generationtimeMs": features.get("source_generationtime_ms"),
                    "timezone": features.get("source_timezone"),
                    "runId": None,
                    "runIdNote": "The standard source endpoint does not expose a stable forecast-run ID.",
                    # `fetchedAt` (and the "data age" hard gate computed from
                    # it) measures when *this server* retrieved the response,
                    # not when the underlying NBM/HRRR/GFS model run was
                    # actually produced. A model run from hours ago fetched
                    # just now reads as age-zero. Without a real run
                    # identifier this cannot be verified from this endpoint;
                    # see AUDIT.md item 14.
                    "sourceRunAgeVerified": False,
                },
                "settlementSource": "NOAA/NCEI Daily Summaries TMAX (final); NWS observations are preliminary",
                "dataQualityStatus": status.value,
                "currentObservedTemperatureF": round(float(current_temperature), 1) if current_temperature is not None else None,
                "intradayObservations": observation.get("observations", []) if observation else [],
                "highF": high,
                "rawModelHighF": raw_model_high_rounded,
                "baselineHighF": baseline_high_rounded,
                # The displayed high can be stabilized or lifted by today's
                # observation. Compare NBM with the raw model output instead.
                "modelDeltaF": raw_model_high_rounded - baseline_high_rounded,
                # Preserve the evaluated model interval. Display smoothing
                # has no validated uncertainty translation policy.
                "rangeLowF": int(np.floor(interval_lower)) if interval_lower is not None else None,
                "rangeHighF": int(np.ceil(interval_upper)) if interval_upper is not None else None,
                "fourDegreeRangeLowF": high - FOUR_DEGREE_HALF_WIDTH_F if is_calibrated else None,
                "fourDegreeRangeHighF": high + FOUR_DEGREE_HALF_WIDTH_F if is_calibrated else None,
                "uncertainty": "Low" if half_width is not None and half_width <= 2.5 else ("Moderate" if half_width is not None else "Unavailable"),
                "modelRange": [round(float(prediction["interval_lower_f"]), 1), round(float(prediction["interval_upper_f"]), 1)] if prediction is not None else None,
                "observedHighSoFarF": observed_high_rounded,
                "observedLowSoFarF": int(np.rint(observed_low)) if observed_low is not None else None,
                "lastObservationAt": observation.get("lastSuccessfulObservationTime") if observation else None,
                "dataFreshness": round(freshness, 2),
                "sourceAgreement": round(source_agreement, 2),
                "reasonCodes": tuple(filter(None, (
                    f"Model-source spread is {disagreement_f:.1f}°F." if np.isfinite(disagreement_f) else None,
                    "Current station observation is below the model estimate." if current_temperature is not None and current_temperature < predicted_high else None,
                    "Observed high has been incorporated as a floor." if observed_high is not None else None,
                    "Live NBM baseline shown; station-specific residual calibration is not yet available." if not is_calibrated else None,
                    "NO BET: complete NBM, HRRR, and GFS guidance is required for the evaluated residual-MOS contract." if not complete_guidance else None,
                    f"NO BET: residual MOS is evaluated only at a {SUPPORTED_FORECAST_LEAD_DAYS}-day lead; this request is {lead_days} days." if not supported_horizon else None,
                    f"NO BET: only {completeness:.0%} of the model's expected features are finite (minimum {MIN_PREDICTION_FEATURE_COMPLETENESS:.0%})." if not features_sufficient else None,
                    "NO BET: no provider-specific market settlement contract has been verified for this display." ,
                ))),
                "stabilityReason": stability_reason,
                # The decision layer is intentionally the last thing to run
                # and is kept in its own key: the forecast fields above are
                # exactly what they were before it existed, so the weather
                # model and the betting filter can never be confused for
                # one another by a consumer of this payload.
                "betDecision": _bet_decision(
                    station=station, features=features, observation=observation,
                    predicted_high_f=predicted_high,
                    interval_lower_f=interval_lower, interval_upper_f=interval_upper,
                    is_calibrated=is_calibrated, supported_horizon=supported_horizon,
                    completeness=completeness, target_date=target_date,
                    is_same_day=observation is not None,
                    snapshots=snapshots, config=filter_config,
                ),
            }
        )
    stability_state["forecasts"] = saved_forecasts
    stability_state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return forecasts, stability_state


def _bet_summary(forecasts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Board-level selectivity counts.

    Deliberately reports how much of the board was rejected, not just what
    was recommended: a board showing four plays out of twenty is the
    intended outcome and needs to read that way rather than looking like
    sixteen failures.
    """
    decisions = [item["betDecision"] for item in forecasts if item.get("betDecision")]
    if not decisions:
        return None
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision["tier"]] = counts.get(decision["tier"], 0) + 1
    recommended = [item for item in decisions if item["recommended"]]
    rejected = [item for item in decisions if not item["recommended"]]
    probabilities = [item["bucket"]["probability"] for item in recommended if item.get("bucket")]
    rejected_probabilities = [item["bucket"]["probability"] for item in rejected if item.get("bucket")]
    return {
        "evaluated": len(decisions),
        "counts": counts,
        "recommended": len(recommended),
        "coverageRate": round(len(recommended) / len(decisions), 4),
        "mode": decisions[0]["mode"],
        "averageRecommendedProbability": (
            round(float(np.mean(probabilities)), 4) if probabilities else None
        ),
        "averagePassProbability": (
            round(float(np.mean(rejected_probabilities)), 4) if rejected_probabilities else None
        ),
    }


def _assemble_payload(target_date: date, forecasts: list[dict[str, Any]], model_version: str) -> dict:
    return {
        "targetDate": target_date.isoformat(),
        "today": dashboard_today().isoformat(),
        "maxDate": (dashboard_today() + timedelta(days=7)).isoformat(),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "forecasts": forecasts,
        "stationRegistry": station_metadata(),
        "supportedMarketTypes": ["daily_high", "daily_low", "temperature_bracket", "threshold_gte", "threshold_lte"],
        "trend": _trend(),
        "accuracy": _accuracy(),
        "modelEvidence": _model_evidence(),
        "forecastInputs": "NCEP NBM + HRRR + GFS forecast guidance",
        "validationTarget": "Official NOAA/NCEI daily TMAX",
        "evaluationContract": "10,279 held-out station-day forecasts across all 20 configured stations; archived 24-hour lead composite",
        # The accuracy/trend/betDecision figures above all score the model's
        # raw prediction. `forecasts[].highF` can differ from
        # `forecasts[].rawModelHighF` when `stabilize_display_high` holds a
        # small refresh change or an observed high lifts the display value;
        # no equivalent backtest measures the *displayed* value's own hit
        # rate or interval coverage. See AUDIT.md item 16.
        "evaluationBasis": "Accuracy, trend, and betDecision figures are scored against the raw model prediction (rawModelHighF), not the display-stabilized highF shown on the board.",
        "releaseStatus": "Experimental shadow monitoring — not operational guidance",
        "modelVersion": model_version,
        "stabilityPolicy": "Minor refresh changes under 2°F are held; observed highs can update today.",
        "betSummary": _bet_summary(forecasts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=_date, default=dashboard_today())
    args = parser.parse_args()
    print(json.dumps(payload(args.date)))


if __name__ == "__main__":
    main()
