"""Same-day reconciliation between what was forecast and what is happening.

For a market settling today, the observations already in hand are the
strongest evidence available and they are not optional. A station sitting at
91F at 1 PM when the model expected 89.8F is telling you something the 6 AM
forecast cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeatingCurve:
    """Historical remaining temperature rise by local hour for one station.

    Deliberately per-station: KPHX in September still has eight degrees to
    climb at 10 AM, KSEA does not, and a single shared curve would be wrong
    for both.
    """

    station: str
    month: int
    remaining_by_hour_f: dict[int, float]
    sample_size: int

    def remaining_at(self, local_hour: int) -> float | None:
        if not self.remaining_by_hour_f:
            return None
        if local_hour in self.remaining_by_hour_f:
            return self.remaining_by_hour_f[local_hour]
        hours = sorted(self.remaining_by_hour_f)
        if local_hour < hours[0]:
            return self.remaining_by_hour_f[hours[0]]
        if local_hour > hours[-1]:
            return self.remaining_by_hour_f[hours[-1]]
        below = max(hour for hour in hours if hour <= local_hour)
        above = min(hour for hour in hours if hour >= local_hour)
        if below == above:
            return self.remaining_by_hour_f[below]
        weight = (local_hour - below) / (above - below)
        return float(
            self.remaining_by_hour_f[below] * (1 - weight)
            + self.remaining_by_hour_f[above] * weight
        )


@dataclass(frozen=True)
class ObservationAlignment:
    is_same_day: bool
    current_temperature_f: float | None
    observed_high_so_far_f: float | None
    expected_by_now_f: float | None
    anomaly_f: float | None
    expected_remaining_heating_f: float | None
    implied_high_f: float | None
    alignment_score: float
    bucket_already_impossible: bool
    notes: tuple[str, ...] = ()


def analyze(
    is_same_day: bool,
    predicted_high_f: float,
    current_temperature_f: float | None = None,
    observed_high_so_far_f: float | None = None,
    local_hour: int | None = None,
    heating_curve: HeatingCurve | None = None,
    bucket_upper_f: float | None = None,
    tolerance_f: float = 2.0,
) -> ObservationAlignment:
    """Compare live observations against the forecast's own expectations.

    For a future-dated market there are no observations to reconcile, and
    this returns a neutral score rather than a strong one: today's readings
    say nothing about tomorrow's high and must never be borrowed to support
    it.
    """
    notes: list[str] = []
    if not is_same_day:
        return ObservationAlignment(
            is_same_day=False, current_temperature_f=None, observed_high_so_far_f=None,
            expected_by_now_f=None, anomaly_f=None, expected_remaining_heating_f=None,
            implied_high_f=None, alignment_score=0.5, bucket_already_impossible=False,
            notes=("Future-dated market: no same-day observations apply.",),
        )
    if observed_high_so_far_f is None and current_temperature_f is None:
        return ObservationAlignment(
            is_same_day=True, current_temperature_f=None, observed_high_so_far_f=None,
            expected_by_now_f=None, anomaly_f=None, expected_remaining_heating_f=None,
            implied_high_f=None, alignment_score=0.0, bucket_already_impossible=False,
            notes=("No station observation has been reported for today yet.",),
        )
    remaining = (
        heating_curve.remaining_at(local_hour)
        if heating_curve is not None and local_hour is not None
        else None
    )
    reference = observed_high_so_far_f if observed_high_so_far_f is not None else current_temperature_f
    implied = reference + remaining if (reference is not None and remaining is not None) else None
    expected_by_now = predicted_high_f - remaining if remaining is not None else None
    anomaly = (
        reference - expected_by_now
        if (reference is not None and expected_by_now is not None)
        else None
    )
    # A high already above the bucket's top edge is not a low-probability
    # bet, it is a settled loss.
    impossible = bool(
        observed_high_so_far_f is not None
        and bucket_upper_f is not None
        and observed_high_so_far_f >= bucket_upper_f
    )
    if impossible:
        notes.append(
            f"Observed high {observed_high_so_far_f:.0f}F already exceeds the bucket's upper bound."
        )
    if anomaly is None:
        score = 0.4
        notes.append("No station heating curve available; observation alignment is unverified.")
    else:
        score = float(max(0.0, 1.0 - abs(anomaly) / tolerance_f))
        if anomaly > tolerance_f:
            notes.append(f"Station is running {anomaly:.1f}F hotter than the forecast expected by now.")
        elif anomaly < -tolerance_f:
            notes.append(f"Station is running {abs(anomaly):.1f}F cooler than the forecast expected by now.")
    if impossible:
        score = 0.0
    return ObservationAlignment(
        is_same_day=True,
        current_temperature_f=current_temperature_f,
        observed_high_so_far_f=observed_high_so_far_f,
        expected_by_now_f=expected_by_now,
        anomaly_f=anomaly,
        expected_remaining_heating_f=remaining,
        implied_high_f=implied,
        alignment_score=score,
        bucket_already_impossible=impossible,
        notes=tuple(notes),
    )


def build_heating_curve(
    station: str,
    month: int,
    hourly_observations: dict[int, list[float]],
    daily_highs: list[float],
) -> HeatingCurve:
    """Derive median remaining heating by local hour from station history.

    ``hourly_observations`` maps local hour to that hour's temperatures, and
    ``daily_highs`` holds each day's final high in the same order, so the
    remaining rise is measured per day rather than by differencing two
    independently-computed averages.
    """
    remaining: dict[int, float] = {}
    for hour, temperatures in hourly_observations.items():
        paired = [
            high - temperature
            for temperature, high in zip(temperatures, daily_highs, strict=False)
            if np.isfinite(temperature) and np.isfinite(high)
        ]
        if paired:
            remaining[int(hour)] = float(np.median(paired))
    return HeatingCurve(
        station=station,
        month=month,
        remaining_by_hour_f=remaining,
        sample_size=len(daily_highs),
    )
