"""Forecast-revision tracking.

A forecast that has said 94.5F for nine hours and a forecast that arrived at
94.5F this run after being at 96.2F at dawn are not the same evidence, even
though they print identically. The second one is still moving, and a 2F
market has no room for a forecast that is still moving.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np


@dataclass(frozen=True)
class ForecastSnapshot:
    """One recorded forecast for a station-day, taken at a point in time."""

    captured_at: datetime
    predicted_high_f: float
    bucket_lower_f: int | None = None
    ensemble_spread_f: float | None = None


@dataclass(frozen=True)
class StabilityAnalysis:
    snapshot_count: int
    change_3h_f: float | None
    change_6h_f: float | None
    change_12h_f: float | None
    change_24h_f: float | None
    volatility_f: float
    bucket_flips_12h: int
    trend_direction: str
    trend_consistency: float
    stability_score: float
    hours_observed: float


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _change_over(snapshots: list[ForecastSnapshot], now: datetime, hours: float) -> float | None:
    """Signed movement from the oldest snapshot inside the window to the newest."""
    cutoff = now - timedelta(hours=hours)
    window = [item for item in snapshots if _as_utc(item.captured_at) >= cutoff]
    if len(window) < 2:
        return None
    return float(window[-1].predicted_high_f - window[0].predicted_high_f)


def analyze(
    snapshots: list[ForecastSnapshot],
    now: datetime | None = None,
    reference_volatility_f: float = 2.0,
) -> StabilityAnalysis:
    """Score how settled a forecast is.

    With fewer than two snapshots there is no stability evidence at all. The
    score returned is 0.0, not a neutral 0.5: a brand-new forecast has not
    earned confidence, and treating unknown as average is exactly how
    missing data turns into false confidence.
    """
    ordered = sorted(
        (item for item in snapshots if np.isfinite(item.predicted_high_f)),
        key=lambda item: _as_utc(item.captured_at),
    )
    reference = _as_utc(now) if now else (
        _as_utc(ordered[-1].captured_at) if ordered else datetime.now(timezone.utc)
    )
    if len(ordered) < 2:
        return StabilityAnalysis(
            snapshot_count=len(ordered), change_3h_f=None, change_6h_f=None,
            change_12h_f=None, change_24h_f=None, volatility_f=float("nan"),
            bucket_flips_12h=0, trend_direction="unknown", trend_consistency=0.0,
            stability_score=0.0, hours_observed=0.0,
        )
    values = np.asarray([item.predicted_high_f for item in ordered], dtype=float)
    steps = np.diff(values)
    volatility = float(np.mean(np.abs(steps)))
    recent = [
        item for item in ordered
        if _as_utc(item.captured_at) >= reference - timedelta(hours=12)
    ]
    buckets = [item.bucket_lower_f for item in recent if item.bucket_lower_f is not None]
    flips = sum(1 for before, after in zip(buckets, buckets[1:], strict=False) if before != after)
    net = float(values[-1] - values[0])
    if abs(net) < 0.25:
        direction = "flat"
    else:
        direction = "rising" if net > 0 else "falling"
    # Consistency: how much of the total travel was in one direction. A
    # forecast that moved 2F down in a straight line is more interpretable
    # than one that wandered 2F net across 6F of churn.
    travel = float(np.sum(np.abs(steps)))
    consistency = abs(net) / travel if travel > 1e-9 else 1.0
    hours = (_as_utc(ordered[-1].captured_at) - _as_utc(ordered[0].captured_at)).total_seconds() / 3600.0
    score = max(0.0, 1.0 - volatility / reference_volatility_f)
    if flips:
        # Crossing a bucket boundary is categorically worse than drifting
        # inside one: it means the recommendation itself already changed.
        score *= 0.5 ** flips
    return StabilityAnalysis(
        snapshot_count=len(ordered),
        change_3h_f=_change_over(ordered, reference, 3),
        change_6h_f=_change_over(ordered, reference, 6),
        change_12h_f=_change_over(ordered, reference, 12),
        change_24h_f=_change_over(ordered, reference, 24),
        volatility_f=volatility,
        bucket_flips_12h=flips,
        trend_direction=direction,
        trend_consistency=float(consistency),
        stability_score=float(max(0.0, min(1.0, score))),
        hours_observed=float(hours),
    )
