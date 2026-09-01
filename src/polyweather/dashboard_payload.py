"""Small JSON adapter used by the local Vite dashboard development server."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from .data import fetch_live_forecast_features, fetch_live_station_snapshot
from .markets import QualityStatus, quality_gate
from .model import AdaptiveResidualForecaster, BlendedResidualForecaster, ResidualForecaster
from .stations import STATIONS, station_metadata


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "production_v2" / "xgb_residual_tmax.joblib"
ADAPTIVE_MODEL_PATH = ROOT / "artifacts" / "production" / "adaptive_residual_tmax.joblib"
FALLBACK_MODEL_PATH = ROOT / "artifacts" / "production" / "xgb_residual_tmax.joblib"
PREDICTIONS_PATH = ROOT / "artifacts" / "backtest_v2" / "rolling_predictions.parquet"
STATION_METRICS_PATH = ROOT / "artifacts" / "backtest_v2" / "station_metrics.csv"
OVERALL_METRICS_PATH = ROOT / "artifacts" / "backtest_v2" / "overall_metrics.csv"
DASHBOARD_TIMEZONE = ZoneInfo("America/New_York")
STABILITY_STATE_PATH = ROOT / "data" / "normalized" / "dashboard_forecast_state.json"
MIN_STABLE_CHANGE_F = 2
FOUR_DEGREE_HALF_WIDTH_F = 2
STABILITY_STATE_RETENTION_DAYS = 45


def _date(value: str) -> date:
    return date.fromisoformat(value)


def dashboard_today() -> date:
    return datetime.now(DASHBOARD_TIMEZONE).date()


def validate_dashboard_date(target_date: date) -> None:
    today = dashboard_today()
    if not today <= target_date <= today + timedelta(days=7):
        raise ValueError(
            f"Choose a date from {today.isoformat()} through {(today + timedelta(days=7)).isoformat()}."
        )


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
    """Use the artifact's trained station set, not every configured station."""
    try:
        manifest = json.loads(model_path.with_name("model_manifest.json").read_text(encoding="utf-8"))
        codes = [str(code).upper() for code in manifest["stations"]]
        selected = [STATIONS[code] for code in codes if code in STATIONS]
        if selected:
            return selected
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return list(STATIONS.values())


def payload(target_date: date) -> dict:
    validate_dashboard_date(target_date)
    model_path = MODEL_PATH if MODEL_PATH.exists() else ADAPTIVE_MODEL_PATH if ADAPTIVE_MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    model: ResidualForecaster | AdaptiveResidualForecaster | BlendedResidualForecaster = joblib.load(model_path)
    model_version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    model_stations = _model_stations(model_path)
    trained_station_ids = {station.icao for station in model_stations}
    # The UI needs every configured market.  Untrained stations deliberately
    # receive a live numerical baseline and a NO BET status rather than being
    # hidden or assigned an unvalidated residual-model correction.
    display_stations = list(STATIONS.values())
    stability_state = _load_stability_state()
    _prune_stability_state(stability_state, dashboard_today())
    saved_forecasts: dict[str, Any] = stability_state["forecasts"]
    forecasts = []
    # Network retrieval dominates response time. Each station uses a separate
    # endpoint request, so running these in parallel removes the avoidable
    # five-station waterfall without changing the forecast contract.
    with ThreadPoolExecutor(max_workers=min(8, len(display_stations))) as executor:
        futures = [executor.submit(_fetch_live_inputs, station, target_date) for station in display_stations]
        live_inputs = [future.result() for future in futures]
    for station, features, observation in live_inputs:
        icao = station.icao
        baseline_high = float(features.get("nbm_baseline_f", np.nan))
        is_calibrated = icao in trained_station_ids
        prediction = model.predict(pd.DataFrame([features])).iloc[0] if is_calibrated else None
        predicted_high = float(prediction["prediction_f"]) if prediction is not None else baseline_high
        if not np.isfinite(predicted_high) or not np.isfinite(baseline_high):
            raise ValueError(f"Forecast inputs are incomplete for {icao}; no estimate was displayed.")
        raw_model_high_rounded = int(np.rint(predicted_high))
        baseline_high_rounded = int(np.rint(baseline_high))
        observed_high = observation.get("observedDailyHighF") if observation else None
        observed_low = observation.get("observedDailyLowF") if observation else None
        current_temperature = observation.get("currentObservedTemperatureF") if observation else None
        candidate_high = int(np.rint(max(predicted_high, observed_high or float("-inf"))))
        observed_high_rounded = int(np.rint(observed_high)) if observed_high is not None else None
        # A model upgrade should not inherit a display value held from the
        # previous artifact. Within one artifact, the existing two-degree
        # stability rule still suppresses inconsequential refresh churn.
        state_key = f"{model_version}::{target_date.isoformat()}::{icao}"
        previous = saved_forecasts.get(state_key, {}).get("display_high_f")
        high, stability_reason = stabilize_display_high(previous, candidate_high, observed_high_rounded)
        saved_forecasts[state_key] = {
            "display_high_f": high,
            "candidate_high_f": candidate_high,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        half_width = float(prediction["conformal_halfwidth_f"]) if prediction is not None else 4.0
        disagreement_f = float(features.get("model_agreement__tmax_f_spread", np.nan))
        source_agreement = max(0.0, min(1.0, 1.0 - disagreement_f / 8.0)) if np.isfinite(disagreement_f) else 0.0
        freshness = 0.0 if observation and observation.get("stale") else 1.0
        status = quality_gate(
            station_known=True, rules_available=False, freshness=freshness, agreement=source_agreement,
            interval_width_f=2 * half_width,
        )
        if not is_calibrated:
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
                # Anchored on the stabilized display high, but sized from the
                # model's own per-station split-conformal half-width rather
                # than a fixed placeholder, so the band reflects that
                # station's real calibrated uncertainty for this forecast.
                "rangeLowF": int(np.rint(high - half_width)),
                "rangeHighF": int(np.rint(high + half_width)),
                "fourDegreeRangeLowF": high - FOUR_DEGREE_HALF_WIDTH_F,
                "fourDegreeRangeHighF": high + FOUR_DEGREE_HALF_WIDTH_F,
                "uncertainty": "Low" if half_width <= 2.5 else "Moderate",
                "modelRange": [round(float(prediction["p10_f"]), 1), round(float(prediction["p90_f"]), 1)] if prediction is not None else [round(predicted_high - half_width, 1), round(predicted_high + half_width, 1)],
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
                ))),
                "stabilityReason": stability_reason,
            }
        )
    stability_state["forecasts"] = saved_forecasts
    stability_state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_stability_state(stability_state)
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
        "evaluationContract": "2,459 held-out station-day forecasts; archived 24-hour lead composite",
        "releaseStatus": "Experimental shadow monitoring — not operational guidance",
        "modelVersion": model_version,
        "stabilityPolicy": "Minor refresh changes under 2°F are held; observed highs can update today.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=_date, default=dashboard_today())
    args = parser.parse_args()
    print(json.dumps(payload(args.date)))


if __name__ == "__main__":
    main()
