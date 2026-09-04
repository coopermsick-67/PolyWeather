"""Station reliability with explicit small-sample discipline.

Three wins from three bets is not a 100% station. This module exists mainly
to stop that number from ever reaching a score. Every rate it reports is
shrunk toward a prior whose strength is set by how little evidence exists,
and every rate carries the interval that says how little we actually know.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StationReliability:
    station: str
    resolved_rows: int
    raw_accuracy: float | None
    adjusted_accuracy: float
    accuracy_interval: tuple[float, float]
    mae_f: float | None
    bias_f: float | None
    recommended_bets: int
    recommended_wins: int
    raw_win_rate: float | None
    adjusted_win_rate: float | None
    reliability_confidence: float
    status: str

    @property
    def has_credible_history(self) -> bool:
        return self.status in {"active", "caution", "restricted"}


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct near 0 and 1 where the normal one is not."""
    if trials <= 0:
        return (0.0, 1.0)
    phat = successes / trials
    denominator = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def shrink(successes: int, trials: int, prior_rate: float, prior_rows: int) -> float:
    """Beta-Binomial posterior mean.

    ``prior_rows`` is how many observations the prior is worth. At
    ``trials == prior_rows`` the estimate sits halfway between the station's
    own record and the global rate, and the station's own record only
    dominates once it has clearly outweighed the prior.
    """
    if trials <= 0:
        return prior_rate
    alpha = prior_rate * prior_rows
    beta = (1.0 - prior_rate) * prior_rows
    return float((successes + alpha) / (trials + alpha + beta))


def assess(
    station: str,
    resolved_rows: int,
    exact_bucket_hits: int,
    prior_accuracy: float,
    prior_rows: int,
    min_rows_for_credit: int,
    mae_f: float | None = None,
    bias_f: float | None = None,
    recommended_bets: int = 0,
    recommended_wins: int = 0,
    restricted_below: float = 0.50,
    caution_below: float = 0.58,
) -> StationReliability:
    """Build a station's reliability record.

    A station with too little history is ``building``: it is not blocked,
    but its own numbers are not allowed to *raise* its score either. Cold
    stations lean on ensemble agreement, stability, and boundary safety
    instead -- evidence that does not require a track record.
    """
    raw_accuracy = exact_bucket_hits / resolved_rows if resolved_rows > 0 else None
    adjusted = shrink(exact_bucket_hits, resolved_rows, prior_accuracy, prior_rows)
    interval = wilson_interval(exact_bucket_hits, resolved_rows) if resolved_rows > 0 else (0.0, 1.0)
    raw_win_rate = recommended_wins / recommended_bets if recommended_bets > 0 else None
    adjusted_win_rate = (
        shrink(recommended_wins, recommended_bets, prior_accuracy, prior_rows)
        if recommended_bets > 0
        else None
    )
    # Confidence in the reliability estimate itself, not in the forecast.
    # Saturating rather than linear: the 20th observation informs far more
    # than the 200th.
    reliability_confidence = resolved_rows / (resolved_rows + prior_rows) if resolved_rows > 0 else 0.0
    if resolved_rows < min_rows_for_credit:
        status = "building"
    elif adjusted < restricted_below:
        status = "restricted"
    elif adjusted < caution_below:
        status = "caution"
    else:
        status = "active"
    return StationReliability(
        station=station,
        resolved_rows=resolved_rows,
        raw_accuracy=raw_accuracy,
        adjusted_accuracy=adjusted,
        accuracy_interval=interval,
        mae_f=mae_f,
        bias_f=bias_f,
        recommended_bets=recommended_bets,
        recommended_wins=recommended_wins,
        raw_win_rate=raw_win_rate,
        adjusted_win_rate=adjusted_win_rate,
        reliability_confidence=float(reliability_confidence),
        status=status,
    )


def reliability_component(record: StationReliability, min_rows_for_credit: int) -> float:
    """Map a reliability record onto a 0-1 score component.

    Deliberately asymmetric. A thin record cannot lift the score above the
    neutral point, but a genuinely poor record still drags it down: absence
    of evidence is not evidence of quality, while evidence of failure is
    evidence of failure regardless of sample size.
    """
    neutral = 0.5
    if record.resolved_rows <= 0:
        return neutral * 0.6
    scaled = max(0.0, min(1.0, (record.adjusted_accuracy - 0.40) / 0.40))
    if record.resolved_rows < min_rows_for_credit:
        return min(neutral, scaled)
    return scaled
