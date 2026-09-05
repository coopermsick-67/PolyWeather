"""Research probability estimates for hypothetical integer threshold lines.

A line solved from a distribution is not evidence that a corresponding market
exists or is executable. Neither this helper nor a high historical hit rate
validates the input distribution, its probabilities, or financial returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math
import numpy as np

from .distribution import TemperatureDistribution

Side = Literal["gte", "lte"]


@dataclass(frozen=True)
class ThresholdContract:
    """A settled-probability estimate for one over/under line."""

    side: Side
    threshold_f: int
    win_probability: float
    margin_from_forecast_f: float
    # Restated on every instance because this number reads like an edge and
    # is not one.
    priced_edge_claimed: bool = False

    @property
    def label(self) -> str:
        return f"high {'>=' if self.side == 'gte' else '<='} {self.threshold_f}F"


def _settlement_cutoff(threshold_f: int, side: Side, rounding: str = "nearest") -> float:
    """The continuous temperature at which an integer threshold flips.

    Settlement rounds the observed high before comparing it, so a ">= 95F"
    contract is won by a true high of 94.5F. Comparing against the printed
    integer instead would misprice every threshold by half a degree, the
    same half-degree ``settlement_window`` exists to get right for brackets.
    """
    if rounding == "nearest":
        return threshold_f - 0.5 if side == "gte" else threshold_f + 0.5
    if rounding == "truncate":
        return float(threshold_f) if side == "gte" else float(threshold_f) + 1.0
    if rounding == "exact":
        return float(threshold_f)
    raise ValueError(f"Unknown settlement rounding rule: {rounding}")


def win_probability(
    distribution: TemperatureDistribution,
    threshold_f: int,
    side: Side,
    rounding: str = "nearest",
) -> float:
    """Probability this threshold contract settles in our favour."""
    if side not in ("gte", "lte"):
        raise ValueError(f"Unknown threshold side: {side!r}. Use 'gte' or 'lte'.")
    if isinstance(threshold_f, (bool, np.bool_)) or not math.isfinite(threshold_f) or int(threshold_f) != threshold_f:
        raise ValueError("Threshold must be a finite integer temperature.")
    cutoff = _settlement_cutoff(threshold_f, side, rounding)
    # Nearest/floor LTE loses at the upper half-open cutoff; exact LTE
    # includes equality. GTE always includes equality at its lower cutoff.
    below = distribution.cdf(cutoff if side == "lte" and rounding == "exact"
                             else np.nextafter(cutoff, -np.inf))
    # ``cdf`` already renormalises over an observed-high floor, so a line the
    # day has physically ruled out reports 0 or 1 rather than a stale prior.
    return float(max(0.0, min(1.0, below if side == "lte" else 1.0 - below)))


def contract_for_target(
    distribution: TemperatureDistribution,
    target_probability: float,
    side: Side,
    rounding: str = "nearest",
) -> ThresholdContract:
    """The nearest integer line that still clears ``target_probability``.

    Solves for the line from the calibrated distribution, then walks outward
    to the safe side of the integer rounding. Rounding inward would hand back
    a contract whose true probability sits just under the target the caller
    asked for, which is the failure mode worth engineering against here.
    """
    if not 0.0 < target_probability < 1.0:
        raise ValueError("Target probability must be strictly between 0 and 1.")
    if side not in ("gte", "lte"):
        raise ValueError(f"Unknown threshold side: {side!r}. Use 'gte' or 'lte'.")

    exact = distribution.quantile(
        1.0 - target_probability if side == "gte" else target_probability
    )
    # Step to the integer line on the conservative side of `exact`, then keep
    # stepping while the achieved probability still falls short -- one step is
    # enough for a continuous distribution, but an empirical one is a step
    # function and can need more.
    if not math.isfinite(exact):
        raise ValueError("Cannot solve a threshold from a non-finite quantile.")
    center = math.floor(exact)
    # Search from the most demanding line toward the easier side. This is
    # bounded for BOTH directions and also handles atoms/negative temperatures.
    # Starting only on the easy side of a quantile can skip the nearest valid
    # integer line under nearest-degree settlement.
    candidates = range(center + 40, center - 41, -1) if side == "gte" else range(center - 40, center + 41)
    for candidate in candidates:
        achieved = win_probability(distribution, candidate, side, rounding)
        if achieved >= target_probability:
            return ThresholdContract(
                side=side,
                threshold_f=candidate,
                win_probability=achieved,
                margin_from_forecast_f=float(
                    distribution.expected_high_f - candidate if side == "gte"
                    else candidate - distribution.expected_high_f
                ),
            )
    raise ValueError(
        f"No threshold within 40F reached {target_probability:.0%} for side {side!r}; "
        "the distribution is too wide or an observed-high floor has ruled the side out."
    )
