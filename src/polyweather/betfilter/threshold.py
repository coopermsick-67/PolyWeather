"""Threshold (over/under) contracts, where a high win rate is actually reachable.

A 2F bracket asks the forecast to be almost exact, and held-out measurement
says that lands about 38% of the time -- no amount of selectivity in
``reports/accuracy_improvement_2026-09-05/selective_win_rate.json`` lifted it
past 46%. A threshold contract asks a strictly easier question: not "which
bucket" but "which side of this line". Pushing the line away from the point
forecast buys win rate directly, and on frozen held-out data a 3F margin
settled our way 91.8% (>=) and 93.2% (<=) of the time.

The thing that buys is win *rate*, and nothing else. A contract that settles
our way 92% of the time is priced near 92 cents by anyone competent, so a
high win rate here is not an edge and not profit -- it is the market's own
estimate, restated. Every function below reports probability only; none of
them claims value, and callers must not present these numbers as an edge
without pricing the other side of the trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    cutoff = _settlement_cutoff(threshold_f, side, rounding)
    below = distribution.cdf(cutoff)
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
    candidate = int(exact) if side == "gte" else int(exact) + 1
    while candidate > int(exact) - 40:
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
        candidate += -1 if side == "gte" else 1
    raise ValueError(
        f"No threshold within 40F reached {target_probability:.0%} for side {side!r}; "
        "the distribution is too wide or an observed-high floor has ruled the side out."
    )
