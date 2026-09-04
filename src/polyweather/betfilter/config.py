"""Centralized configuration for the bet-decision layer.

Every threshold the decision engine uses lives here. Nothing in the
forecasting pipeline reads this module: the weather model answers "what is
the high?" and this subsystem answers the separate question "is that answer
trustworthy enough to act on?". Keeping the thresholds in one place is what
makes them backtestable -- `betfilter.backtest` sweeps these values against
historical outcomes rather than anyone hand-picking a number that looked
good once.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Mode = Literal["standard", "conservative", "very_conservative"]


@dataclass(frozen=True)
class ScoringWeights:
    """Bet Quality Score component weights. Must sum to 1.0."""

    range_probability: float = 0.30
    ensemble_agreement: float = 0.15
    forecast_stability: float = 0.15
    boundary_safety: float = 0.10
    station_reliability: float = 0.10
    observation_alignment: float = 0.10
    weather_uncertainty: float = 0.05
    forecast_horizon: float = 0.05

    def total(self) -> float:
        return (
            self.range_probability
            + self.ensemble_agreement
            + self.forecast_stability
            + self.boundary_safety
            + self.station_reliability
            + self.observation_alignment
            + self.weather_uncertainty
            + self.forecast_horizon
        )

    def validate(self) -> None:
        if abs(self.total() - 1.0) > 1e-9:
            raise ValueError(f"Scoring weights must sum to 1.0; got {self.total():.6f}.")


@dataclass(frozen=True)
class HardGates:
    """Conditions that force a PASS regardless of how good the score looks.

    These exist because a weighted average can always be dragged back over a
    threshold by unrelated strengths -- a beautifully stable forecast from
    agreeing models is still unbettable if it sits 0.1F from a bucket edge.
    Gates are evaluated before scoring and cannot be outvoted by it.
    """

    minimum_range_probability: float = 0.58
    # Boundary distance measured in units of the forecast's own standard
    # deviation. Raw degrees are not comparable across stations: 0.5F of
    # headroom is comfortable at a station with 0.6F error and meaningless
    # at one with 2.5F error.
    minimum_normalized_boundary_safety: float = 0.35
    # Spread across the major independent guidance sources.
    maximum_ensemble_spread_f: float = 3.5
    # Absolute forecast movement over the trailing window.
    maximum_forecast_revision_6h_f: float = 2.0
    # How many times the most-likely bucket changed in the trailing window.
    maximum_bucket_flips_12h: int = 1
    # Distinct guidance sources that must have reported for this station.
    minimum_data_sources: int = 2
    # Feature completeness the deployed residual model requires.
    minimum_feature_completeness: float = 0.85
    # Age of the newest supporting observation/guidance, in minutes.
    maximum_data_age_minutes: float = 180.0
    # Gap between the best and second-best bucket. A 45%/43% split is a
    # coin flip wearing a favorite's label.
    minimum_probability_gap: float = 0.08
    # Shrunk (not raw) historical exact-bucket accuracy for the station.
    minimum_station_adjusted_accuracy: float = 0.45


@dataclass(frozen=True)
class TierThresholds:
    """Bet Quality Score cutoffs. Applied only after every hard gate passes."""

    elite: float = 90.0
    strong: float = 82.0
    playable: float = 75.0
    marginal: float = 68.0


@dataclass(frozen=True)
class BetFilterConfig:
    enabled: bool = True
    mode: Mode = "conservative"
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    gates: HardGates = field(default_factory=HardGates)
    tiers: TierThresholds = field(default_factory=TierThresholds)
    # Settlement rounding: an observed high of 94.6F settles the 95-96 bucket
    # when settlement rounds to the nearest whole degree. Getting this wrong
    # silently shifts every bucket probability by half a degree.
    settlement_rounding: Literal["nearest", "truncate", "exact"] = "nearest"
    # Beta-Binomial prior strength for station reliability shrinkage. A
    # station with 3 wins from 3 bets must not read as 100% reliable.
    reliability_prior_rows: int = 40
    reliability_prior_accuracy: float = 0.55
    # Minimum resolved rows before a station's own history is allowed to
    # *raise* (rather than only lower) its score.
    reliability_min_rows_for_credit: int = 25
    # Measured on held-out history, this system's raw bucket probabilities
    # run about 4 points overconfident, and up to 13 points in the 60-70%
    # band that the gates actually care about. Applying the isotonic
    # correction is what makes a "63% gate" select markets that really do
    # settle 63% of the time. Disable only to reproduce pre-calibration
    # numbers; never to make the board look more confident.
    apply_probability_calibration: bool = True

    def __post_init__(self) -> None:
        self.weights.validate()

    def for_mode(self, mode: Mode) -> "BetFilterConfig":
        """Return this config re-tightened for a selectivity mode.

        Only decision thresholds move. The weather forecast, its
        distribution, and every measured component are identical across
        modes -- selectivity changes what we are willing to act on, never
        what we believe the weather will be.
        """
        if mode == "standard":
            gates = replace(self.gates, minimum_range_probability=0.58)
            tiers = replace(self.tiers, playable=75.0, marginal=68.0)
        elif mode == "conservative":
            gates = replace(
                self.gates,
                minimum_range_probability=0.63,
                minimum_normalized_boundary_safety=0.45,
                maximum_ensemble_spread_f=3.0,
                maximum_forecast_revision_6h_f=1.75,
            )
            tiers = replace(self.tiers, playable=82.0, marginal=75.0)
        elif mode == "very_conservative":
            gates = replace(
                self.gates,
                minimum_range_probability=0.68,
                minimum_normalized_boundary_safety=0.60,
                maximum_ensemble_spread_f=2.5,
                maximum_forecast_revision_6h_f=1.25,
                maximum_bucket_flips_12h=0,
                minimum_probability_gap=0.15,
                minimum_station_adjusted_accuracy=0.55,
            )
            tiers = replace(self.tiers, playable=88.0, marginal=82.0)
        else:  # pragma: no cover - Mode is a closed Literal.
            raise ValueError(f"Unknown selectivity mode: {mode}")
        return replace(self, mode=mode, gates=gates, tiers=tiers)


DEFAULT_CONFIG = BetFilterConfig().for_mode("conservative")
