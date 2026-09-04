"""Bridge from this repo's evaluation artifacts to bet-filter inputs.

``polyweather.betfilter`` is deliberately pure -- it knows nothing about
file paths or this project's directory layout. This module is the adapter:
it reads the held-out rolling backtest and turns it into the station
reliability records and empirical residual distributions the filter needs.

Everything here comes from ``rolling_predictions.parquet``, which contains
only predictions made by a model fitted strictly before each row's target
date. That property is what makes these numbers usable as priors at all; if
the backtest ever stops being rolling-origin, every reliability figure
downstream becomes leakage.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from .betfilter import StationReliability, assess_reliability
from .betfilter.results import settled_in_bucket

ROOT = Path(__file__).resolve().parents[2]
# Must track dashboard_payload's evidence family: reliability priors derived
# from one model generation do not describe a different one.
PREDICTIONS_PATH = ROOT / "artifacts" / "backtest_v4" / "rolling_predictions.parquet"
CALIBRATOR_PATH = ROOT / "artifacts" / "production_v4" / "bucket_probability_calibrator.json"
CANDIDATE_COLUMN = "xgb_prediction_f"
# The public boards this app tracks use even-anchored two-degree buckets
# (94-95, 96-97). A market on a different grid must pass its own anchor
# through rather than inheriting this default.
DEFAULT_BUCKET_ANCHOR_PARITY = 0
DEFAULT_BUCKET_WIDTH_F = 2


def bucket_for(prediction_f: float, width_f: int = DEFAULT_BUCKET_WIDTH_F, parity: int = DEFAULT_BUCKET_ANCHOR_PARITY) -> tuple[int, int]:
    """The market bucket a point forecast would have selected."""
    rounded = int(np.rint(prediction_f))
    offset = (rounded - parity) % width_f
    lower = rounded - offset
    return (lower, lower + width_f - 1)


@functools.lru_cache(maxsize=1)
def _rolling_predictions() -> pd.DataFrame:
    frame = pd.read_parquet(PREDICTIONS_PATH, columns=["station", "target_date", "tmax_f", CANDIDATE_COLUMN])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame.dropna(subset=["tmax_f", CANDIDATE_COLUMN]).sort_values("target_date")


@functools.lru_cache(maxsize=1)
def station_residuals() -> dict[str, np.ndarray]:
    """Per-station held-out residuals (actual minus predicted), in degrees F.

    These drive the empirical predictive distribution, which is preferred
    over any parametric fit because it carries the real skew and fat tails
    of that station's errors instead of assuming them away.
    """
    frame = _rolling_predictions()
    residual = frame["tmax_f"].to_numpy(float) - frame[CANDIDATE_COLUMN].to_numpy(float)
    output: dict[str, np.ndarray] = {}
    for station, group in frame.assign(residual=residual).groupby("station"):
        values = group["residual"].to_numpy(float)
        output[str(station)] = values[np.isfinite(values)]
    return output


@functools.lru_cache(maxsize=1)
def _exact_bucket_history() -> pd.DataFrame:
    """Would the model's favored bucket have actually settled? Per station."""
    frame = _rolling_predictions().copy()
    buckets = [bucket_for(value) for value in frame[CANDIDATE_COLUMN].to_numpy(float)]
    frame["bucket_lower_f"] = [item[0] for item in buckets]
    frame["bucket_upper_f"] = [item[1] for item in buckets]
    frame["hit"] = [
        settled_in_bucket(actual, lower, upper)
        for actual, lower, upper in zip(
            frame["tmax_f"].to_numpy(float),
            frame["bucket_lower_f"].to_numpy(int),
            frame["bucket_upper_f"].to_numpy(int),
            strict=True,
        )
    ]
    return frame


@functools.lru_cache(maxsize=1)
def global_exact_bucket_accuracy() -> float:
    """The prior that thin-history stations are shrunk toward."""
    frame = _exact_bucket_history()
    return float(frame["hit"].mean()) if len(frame) else 0.5


@functools.lru_cache(maxsize=1)
def station_reliability_records() -> dict[str, StationReliability]:
    """Reliability for every station with held-out evaluation history.

    ``exact bucket accuracy`` here is the metric that actually matters for a
    2F market and is materially lower than the familiar "within 2F" figure:
    landing within two degrees of the truth is not the same as landing in
    the one bucket that pays.
    """
    frame = _exact_bucket_history()
    prior = global_exact_bucket_accuracy()
    records: dict[str, StationReliability] = {}
    for station, group in frame.groupby("station"):
        errors = group[CANDIDATE_COLUMN].to_numpy(float) - group["tmax_f"].to_numpy(float)
        records[str(station)] = assess_reliability(
            station=str(station),
            resolved_rows=int(len(group)),
            exact_bucket_hits=int(group["hit"].sum()),
            prior_accuracy=prior,
            prior_rows=40,
            min_rows_for_credit=25,
            mae_f=float(np.mean(np.abs(errors))),
            bias_f=float(np.mean(errors)),
        )
    return records


def reliability_for(station: str) -> StationReliability:
    """Reliability record for one station, or an explicit cold-start record.

    An unknown station gets zero resolved rows rather than the global
    average. It is a station we have never evaluated, and saying so keeps it
    out of the scoring credit it has not earned.
    """
    try:
        records = station_reliability_records()
        prior = global_exact_bucket_accuracy()
    except (OSError, ValueError, KeyError):
        records, prior = {}, 0.5
    if station in records:
        return records[station]
    return assess_reliability(
        station=station, resolved_rows=0, exact_bucket_hits=0,
        prior_accuracy=prior, prior_rows=40, min_rows_for_credit=25,
    )


def residuals_for(station: str) -> np.ndarray | None:
    try:
        return station_residuals().get(station)
    except (OSError, ValueError, KeyError):
        return None


@functools.lru_cache(maxsize=1)
def probability_calibrator() -> "ProbabilityCalibrator":
    """The fitted bucket-probability correction, or the identity if absent.

    Falling back to the identity is the safe direction: an uncorrected
    probability is merely overconfident, whereas a stale correction fitted
    against a different model generation would be actively wrong.
    """
    from .betfilter.calibration import IDENTITY, ProbabilityCalibrator

    try:
        return ProbabilityCalibrator.load(CALIBRATOR_PATH)
    except (OSError, ValueError, KeyError):
        return IDENTITY


def station_accuracy_table() -> list[dict[str, object]]:
    """Station ranking on exact-bucket accuracy, for the UI and reports."""
    records = station_reliability_records()
    rows = [
        {
            "station": record.station,
            "resolvedDays": record.resolved_rows,
            "rawExactBucketAccuracy": None if record.raw_accuracy is None else round(record.raw_accuracy, 4),
            "adjustedExactBucketAccuracy": round(record.adjusted_accuracy, 4),
            "accuracyIntervalLow": round(record.accuracy_interval[0], 4),
            "accuracyIntervalHigh": round(record.accuracy_interval[1], 4),
            "maeF": None if record.mae_f is None else round(record.mae_f, 2),
            "biasF": None if record.bias_f is None else round(record.bias_f, 2),
            "status": record.status,
        }
        for record in records.values()
    ]
    return sorted(rows, key=lambda row: row["adjustedExactBucketAccuracy"], reverse=True)
