"""Outcome classification and loss attribution.

The platform's own "Win" label is not the metric that matters. A $3.60 entry
returning $2.56 is displayed as a win and is a loss of $1.04. Strategy
performance is measured on money, so ``financial_result`` -- not the
platform label -- is what every reliability and backtest number in this
package consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Result = Literal["WIN", "LOSS"]

LossReason = Literal[
    "OVER_PREDICTED_TEMP",
    "UNDER_PREDICTED_TEMP",
    "BUCKET_BOUNDARY_MISS",
    "FORECAST_CHANGED_LATE",
    "STATION_BIAS",
    "MODEL_DISAGREEMENT_IGNORED",
    "BAD_OBSERVATION_ALIGNMENT",
    "LOW_CONFIDENCE_BET",
    "UNKNOWN",
]

# Inside this distance from a bucket edge, a miss is a rounding-scale event
# rather than a forecasting failure -- a different problem needing a
# different fix, so it is classified separately.
BOUNDARY_MISS_TOLERANCE_F = 1.0


def financial_result(buy_in: float, payout: float) -> Result:
    """A bet is a win only when more money came back than went in.

    Break-even is a loss: capital was risked and nothing was gained.
    """
    if not (np.isfinite(buy_in) and np.isfinite(payout)):
        raise ValueError("Both buy-in and payout must be finite to classify a result.")
    if buy_in < 0 or payout < 0:
        raise ValueError("Buy-in and payout cannot be negative.")
    return "WIN" if payout > buy_in else "LOSS"


def net_profit(buy_in: float, payout: float) -> float:
    return float(payout - buy_in)


@dataclass(frozen=True)
class ResolvedMarket:
    station: str
    target_date: str
    bucket_lower_f: int
    bucket_upper_f: int
    predicted_high_f: float
    actual_high_f: float
    settled_in_bucket: bool
    financial: Result | None = None
    net_profit: float | None = None
    loss_reason: LossReason | None = None


def settled_in_bucket(
    actual_high_f: float, bucket_lower_f: int, bucket_upper_f: int, rounding: str = "nearest"
) -> bool:
    """Whether an observed high settles into a bucket under the stated rule."""
    from .distribution import settlement_window

    window_lower, window_upper = settlement_window(bucket_lower_f, bucket_upper_f, rounding)
    return bool(window_lower <= actual_high_f < window_upper)


def classify_loss(
    predicted_high_f: float,
    actual_high_f: float,
    bucket_lower_f: int,
    bucket_upper_f: int,
    forecast_change_6h_f: float | None = None,
    ensemble_spread_f: float | None = None,
    observation_anomaly_f: float | None = None,
    bucket_probability: float | None = None,
    rounding: str = "nearest",
) -> LossReason:
    """Attribute a loss to its most likely cause.

    A 1F boundary miss and a 4F forecast miss both read as "loss" and need
    completely different fixes; without attribution the aggregate win rate
    cannot tell you which one to work on.
    """
    from .distribution import settlement_window

    window_lower, window_upper = settlement_window(bucket_lower_f, bucket_upper_f, rounding)
    if window_lower <= actual_high_f < window_upper:
        return "UNKNOWN"
    miss_distance = (
        window_lower - actual_high_f if actual_high_f < window_lower else actual_high_f - window_upper
    )
    if miss_distance <= BOUNDARY_MISS_TOLERANCE_F:
        return "BUCKET_BOUNDARY_MISS"
    if forecast_change_6h_f is not None and abs(forecast_change_6h_f) >= 1.5:
        return "FORECAST_CHANGED_LATE"
    if ensemble_spread_f is not None and ensemble_spread_f >= 2.5:
        return "MODEL_DISAGREEMENT_IGNORED"
    if observation_anomaly_f is not None and abs(observation_anomaly_f) >= 2.0:
        return "BAD_OBSERVATION_ALIGNMENT"
    if bucket_probability is not None and bucket_probability < 0.60:
        return "LOW_CONFIDENCE_BET"
    return "OVER_PREDICTED_TEMP" if predicted_high_f > actual_high_f else "UNDER_PREDICTED_TEMP"


def resolve(
    station: str,
    target_date: str,
    bucket_lower_f: int,
    bucket_upper_f: int,
    predicted_high_f: float,
    actual_high_f: float,
    buy_in: float | None = None,
    payout: float | None = None,
    rounding: str = "nearest",
    **classification: object,
) -> ResolvedMarket:
    """Score one resolved market, with money as the arbiter when it exists."""
    hit = settled_in_bucket(actual_high_f, bucket_lower_f, bucket_upper_f, rounding)
    financial: Result | None = None
    profit: float | None = None
    if buy_in is not None and payout is not None:
        financial = financial_result(buy_in, payout)
        profit = net_profit(buy_in, payout)
    reason = None
    if not hit or (financial == "LOSS"):
        reason = classify_loss(
            predicted_high_f, actual_high_f, bucket_lower_f, bucket_upper_f,
            forecast_change_6h_f=classification.get("forecast_change_6h_f"),  # type: ignore[arg-type]
            ensemble_spread_f=classification.get("ensemble_spread_f"),  # type: ignore[arg-type]
            observation_anomaly_f=classification.get("observation_anomaly_f"),  # type: ignore[arg-type]
            bucket_probability=classification.get("bucket_probability"),  # type: ignore[arg-type]
            rounding=rounding,
        )
    return ResolvedMarket(
        station=station, target_date=target_date, bucket_lower_f=bucket_lower_f,
        bucket_upper_f=bucket_upper_f, predicted_high_f=predicted_high_f,
        actual_high_f=actual_high_f, settled_in_bucket=hit, financial=financial,
        net_profit=profit, loss_reason=reason,
    )
