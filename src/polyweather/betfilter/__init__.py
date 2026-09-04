"""Bet-decision layer sitting after the forecast engine.

The forecasting pipeline (``polyweather.model``, ``polyweather.data``)
answers "what will the high be?". This package answers the separate,
narrower question "is that answer good enough to bet on this exact 2F
bucket?" -- and is expected to answer no most of the time.

The two must stay separate. A filter that can edit the forecast will
eventually be tuned to make the forecast look good, which is the failure
mode this whole layer exists to prevent.

    raw guidance -> forecast engine -> temperature distribution
      -> bucket probabilities -> reliability + stability + agreement
      -> bet quality score -> hard rejection gates -> decision
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG, BetFilterConfig, HardGates, ScoringWeights, TierThresholds
from .decision import (
    RECOMMENDED_TIERS,
    TIER_LABELS,
    BetDecision,
    BetEvidence,
    DataQuality,
    decide,
    summarize,
)
from .distribution import (
    BucketProbability,
    TemperatureDistribution,
    candidate_buckets,
    from_calibrated_interval,
    from_residual_history,
    normalized_entropy,
    probability_gap,
    settlement_window,
)
from .ensemble import EnsembleAnalysis
from .ensemble import analyze as analyze_ensemble
from .observation import HeatingCurve, ObservationAlignment
from .observation import analyze as analyze_observation
from .reasons import Reason, ReasonLog
from .reliability import StationReliability, assess as assess_reliability, shrink, wilson_interval
from .results import financial_result, net_profit, resolve, settled_in_bucket
from .stability import ForecastSnapshot, StabilityAnalysis
from .stability import analyze as analyze_stability

__all__ = [
    "DEFAULT_CONFIG",
    "RECOMMENDED_TIERS",
    "TIER_LABELS",
    "BetDecision",
    "BetEvidence",
    "BetFilterConfig",
    "BucketProbability",
    "DataQuality",
    "EnsembleAnalysis",
    "ForecastSnapshot",
    "HardGates",
    "HeatingCurve",
    "ObservationAlignment",
    "Reason",
    "ReasonLog",
    "ScoringWeights",
    "StabilityAnalysis",
    "StationReliability",
    "TemperatureDistribution",
    "TierThresholds",
    "analyze_ensemble",
    "analyze_observation",
    "analyze_stability",
    "assess_reliability",
    "candidate_buckets",
    "decide",
    "financial_result",
    "from_calibrated_interval",
    "from_residual_history",
    "net_profit",
    "normalized_entropy",
    "probability_gap",
    "resolve",
    "settled_in_bucket",
    "settlement_window",
    "shrink",
    "summarize",
    "wilson_interval",
]
