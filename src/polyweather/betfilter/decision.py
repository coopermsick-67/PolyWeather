"""The decision engine: does this forecast justify a bet at all?

This module never forecasts anything. It receives a finished forecast plus
the evidence around it and answers a separate question: is that forecast
trustworthy enough, for this specific 2F bucket, to act on? PASS is a
first-class answer here, not a failure to produce one.

Order matters. Hard gates run first and cannot be outvoted by the score,
because a weighted average will always let unrelated strengths drag a fatal
weakness back over the line. Only evidence that survives every gate is
scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
from typing import Literal

import numpy as np

from .calibration import IDENTITY, ProbabilityCalibrator
from .config import DEFAULT_CONFIG, BetFilterConfig
from .distribution import (
    BucketProbability,
    TemperatureDistribution,
    candidate_buckets,
    normalized_entropy,
    probability_gap,
)
from .ensemble import EnsembleAnalysis
from .observation import ObservationAlignment
from .reasons import ReasonLog
from .reliability import StationReliability, reliability_component
from .stability import StabilityAnalysis

Tier = Literal["ELITE", "STRONG", "PLAYABLE", "MARGINAL", "PASS", "DATA_INSUFFICIENT"]

TIER_LABELS: dict[str, str] = {
    "ELITE": "ELITE PLAY",
    "STRONG": "STRONG PLAY",
    "PLAYABLE": "PLAYABLE",
    "MARGINAL": "MARGINAL - AVOID",
    "PASS": "PASS",
    "DATA_INSUFFICIENT": "DATA INSUFFICIENT",
}

# Only these two are bets. MARGINAL is deliberately outside the set: it
# exists to be shown and skipped, not to be quietly treated as playable.
RECOMMENDED_TIERS = {"ELITE", "STRONG"}


@dataclass(frozen=True)
class DataQuality:
    """Everything that determines whether we may compute a decision at all."""

    is_calibrated: bool
    supported_horizon: bool
    feature_completeness: float
    source_count: int
    settlement_station_verified: bool
    data_age_minutes: float | None
    forecast_horizon_hours: float | None = None


@dataclass(frozen=True)
class BetEvidence:
    station: str
    target_date: str
    is_same_day: bool
    distribution: TemperatureDistribution
    ensemble: EnsembleAnalysis
    stability: StabilityAnalysis
    reliability: StationReliability
    observation: ObservationAlignment
    data_quality: DataQuality
    market_bucket: tuple[int, int] | None = None
    bucket_width_f: int = 2
    # Fitted on resolved markets that closed before this one. Left as the
    # identity when there is not enough history to support a correction.
    calibrator: ProbabilityCalibrator = IDENTITY


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    raw: float
    weight: float
    weighted_points: float
    maximum_points: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "raw": round(self.raw, 4),
            "points": round(self.weighted_points, 2),
            "maxPoints": round(self.maximum_points, 2),
        }


@dataclass(frozen=True)
class BetDecision:
    station: str
    target_date: str
    tier: Tier
    label: str
    recommended: bool
    quality_score: float
    selected_bucket: BucketProbability | None
    bucket_probability: float | None
    probability_gap: float
    distribution_entropy: float
    components: list[ScoreComponent] = field(default_factory=list)
    reasons: list[dict[str, object]] = field(default_factory=list)
    alternatives: list[BucketProbability] = field(default_factory=list)
    mode: str = "conservative"

    def to_dict(self) -> dict[str, object]:
        bucket = self.selected_bucket
        return {
            "station": self.station,
            "targetDate": self.target_date,
            "tier": self.tier,
            "label": self.label,
            "recommended": self.recommended,
            "qualityScore": round(self.quality_score, 1),
            "mode": self.mode,
            "bucket": None if bucket is None else {
                "lowerF": bucket.lower_f,
                "upperF": bucket.upper_f,
                "label": bucket.label,
                "probability": round(bucket.probability, 4),
                "settlementWindowF": [bucket.window_lower_f, bucket.window_upper_f],
                "minimumEdgeDistanceF": round(bucket.minimum_edge_distance_f, 2),
                "normalizedBoundarySafety": round(bucket.normalized_boundary_safety, 3),
            },
            "probabilityGap": round(self.probability_gap, 4),
            "distributionEntropy": round(self.distribution_entropy, 4),
            "components": [component.to_dict() for component in self.components],
            "reasons": self.reasons,
            "alternatives": [
                {"label": item.label, "probability": round(item.probability, 4)}
                for item in self.alternatives[:4]
            ],
        }


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _probability_component(probability: float) -> float:
    """A coin flip scores zero.

    Mapping raw probability straight onto the score would hand ~50 points
    to a 50% bucket, which is not evidence of anything. The component only
    starts earning above 0.45 and saturates at 0.85, where a 2F market is
    about as good as this data can honestly support.
    """
    return _clip01((probability - 0.45) / 0.40)


def _boundary_component(normalized_safety: float) -> float:
    """One standard deviation of headroom to the nearest edge is full marks."""
    return _clip01(normalized_safety)


def _horizon_component(hours: float | None) -> float:
    """Nearer markets score higher; beyond ~30 hours the edge is gone."""
    if hours is None or not np.isfinite(hours):
        return 0.3
    return _clip01(1.0 - (hours - 3.0) / 27.0)


def score(evidence: BetEvidence, bucket: BucketProbability, config: BetFilterConfig) -> tuple[float, list[ScoreComponent]]:
    """Compute the 0-100 Bet Quality Score and its full breakdown."""
    weights = config.weights
    entropy = 0.0
    raw_values = {
        "Range probability": (_probability_component(bucket.probability), weights.range_probability),
        "Ensemble agreement": (_clip01(evidence.ensemble.agreement_score), weights.ensemble_agreement),
        "Forecast stability": (_clip01(evidence.stability.stability_score), weights.forecast_stability),
        "Boundary safety": (_boundary_component(bucket.normalized_boundary_safety), weights.boundary_safety),
        "Station reliability": (
            reliability_component(evidence.reliability, config.reliability_min_rows_for_credit),
            weights.station_reliability,
        ),
        "Observation alignment": (_clip01(evidence.observation.alignment_score), weights.observation_alignment),
        "Weather uncertainty": (0.0, weights.weather_uncertainty),
        "Forecast horizon": (
            _horizon_component(evidence.data_quality.forecast_horizon_hours),
            weights.forecast_horizon,
        ),
    }
    components: list[ScoreComponent] = []
    total = 0.0
    for name, (raw, weight) in raw_values.items():
        points = 100.0 * weight * raw
        total += points
        components.append(
            ScoreComponent(name=name, raw=raw, weight=weight, weighted_points=points, maximum_points=100.0 * weight)
        )
    return total, components


def _run_hard_gates(
    evidence: BetEvidence,
    bucket: BucketProbability | None,
    gap: float,
    config: BetFilterConfig,
    log: ReasonLog,
) -> Tier | None:
    """Return a terminal tier when any gate fails, otherwise ``None``.

    Data-availability failures return DATA_INSUFFICIENT rather than PASS:
    "we cannot tell" and "we looked and it is not good enough" are different
    answers and collapsing them hides which one happened.
    """
    quality = evidence.data_quality
    gates = config.gates
    if not quality.settlement_station_verified:
        log.add("SETTLEMENT_STATION_UNVERIFIED", "critical",
                "The settlement station for this market has not been verified.")
        return "DATA_INSUFFICIENT"
    if not quality.is_calibrated:
        log.add("UNCALIBRATED_STATION", "critical",
                "This station has no validated residual calibration for the live inputs.")
        return "DATA_INSUFFICIENT"
    if not quality.supported_horizon:
        log.add("UNSUPPORTED_HORIZON", "critical",
                "The model is evaluated only at its supported forecast lead; this request is outside it.")
        return "DATA_INSUFFICIENT"
    if quality.feature_completeness < gates.minimum_feature_completeness:
        log.add("LOW_FEATURE_COMPLETENESS", "critical",
                f"Only {quality.feature_completeness:.0%} of required model inputs are present.",
                quality.feature_completeness, gates.minimum_feature_completeness)
        return "DATA_INSUFFICIENT"
    if quality.source_count < gates.minimum_data_sources:
        log.add("INSUFFICIENT_DATA_SOURCES", "critical",
                f"Only {quality.source_count} guidance source(s) reported; agreement cannot be assessed.",
                quality.source_count, gates.minimum_data_sources)
        return "DATA_INSUFFICIENT"
    if quality.data_age_minutes is not None and quality.data_age_minutes > gates.maximum_data_age_minutes:
        log.add("STALE_DATA", "critical",
                f"Supporting data is {quality.data_age_minutes:.0f} minutes old.",
                quality.data_age_minutes, gates.maximum_data_age_minutes)
        return "DATA_INSUFFICIENT"
    if bucket is None:
        log.add("MISSING_CRITICAL_DATA", "critical", "No market bucket could be evaluated.")
        return "DATA_INSUFFICIENT"

    if evidence.observation.bucket_already_impossible:
        log.add("BUCKET_ALREADY_IMPOSSIBLE", "critical",
                "Today's observed high has already moved past this bucket; it cannot settle here.")
        return "PASS"
    if bucket.probability < gates.minimum_range_probability:
        log.add("LOW_RANGE_PROBABILITY", "critical",
                f"Bucket probability is only {bucket.probability:.1%}.",
                bucket.probability, gates.minimum_range_probability)
        return "PASS"
    if gap < gates.minimum_probability_gap:
        log.add("LOW_PROBABILITY_GAP", "critical",
                f"The favorite leads the next bucket by only {gap:.1%}; the distribution is not concentrated.",
                gap, gates.minimum_probability_gap)
        return "PASS"
    if bucket.normalized_boundary_safety < gates.minimum_normalized_boundary_safety:
        log.add("HIGH_BOUNDARY_RISK", "critical",
                f"Forecast sits {bucket.minimum_edge_distance_f:.1f}F from the nearest bucket edge "
                f"({bucket.normalized_boundary_safety:.2f} sigma).",
                bucket.normalized_boundary_safety, gates.minimum_normalized_boundary_safety)
        return "PASS"
    if np.isfinite(evidence.ensemble.spread_f) and evidence.ensemble.spread_f > gates.maximum_ensemble_spread_f:
        log.add("HIGH_ENSEMBLE_SPREAD", "critical",
                f"Guidance sources span {evidence.ensemble.spread_f:.1f}F.",
                evidence.ensemble.spread_f, gates.maximum_ensemble_spread_f)
        return "PASS"
    change_6h = evidence.stability.change_6h_f
    if change_6h is not None and abs(change_6h) > gates.maximum_forecast_revision_6h_f:
        log.add("FORECAST_UNSTABLE", "critical",
                f"Forecast moved {change_6h:+.1f}F in the last 6 hours.",
                abs(change_6h), gates.maximum_forecast_revision_6h_f)
        return "PASS"
    if evidence.stability.bucket_flips_12h > gates.maximum_bucket_flips_12h:
        log.add("BUCKET_FLIP_RISK", "critical",
                f"The favored bucket changed {evidence.stability.bucket_flips_12h} times in 12 hours.",
                evidence.stability.bucket_flips_12h, gates.maximum_bucket_flips_12h)
        return "PASS"
    reliability = evidence.reliability
    if (
        reliability.resolved_rows >= config.reliability_min_rows_for_credit
        and reliability.adjusted_accuracy < gates.minimum_station_adjusted_accuracy
    ):
        log.add("LOW_STATION_RELIABILITY", "critical",
                f"{reliability.station} historical exact-bucket accuracy is {reliability.adjusted_accuracy:.0%} "
                f"across {reliability.resolved_rows} resolved days.",
                reliability.adjusted_accuracy, gates.minimum_station_adjusted_accuracy)
        return "PASS"
    return None


def _log_supporting_evidence(evidence: BetEvidence, bucket: BucketProbability, log: ReasonLog) -> None:
    if evidence.ensemble.spread_f <= 1.0 and evidence.ensemble.source_count >= 3:
        log.add("HIGH_ENSEMBLE_AGREEMENT", "positive",
                f"{evidence.ensemble.source_count} guidance sources agree within {evidence.ensemble.spread_f:.1f}F.",
                evidence.ensemble.spread_f)
    if evidence.stability.stability_score >= 0.75 and evidence.stability.snapshot_count >= 3:
        log.add("STABLE_FORECAST", "positive",
                f"Forecast has held within {evidence.stability.volatility_f:.1f}F per update "
                f"over {evidence.stability.hours_observed:.0f} hours.",
                evidence.stability.stability_score)
    if bucket.normalized_boundary_safety >= 0.8:
        log.add("SAFE_BUCKET_POSITION", "positive",
                f"Forecast sits {bucket.minimum_edge_distance_f:.1f}F inside the nearest edge.",
                bucket.normalized_boundary_safety)
    if evidence.reliability.status == "active" and evidence.reliability.adjusted_accuracy >= 0.62:
        log.add("STRONG_STATION_HISTORY", "positive",
                f"{evidence.reliability.station} hits its exact bucket {evidence.reliability.adjusted_accuracy:.0%} "
                f"of the time across {evidence.reliability.resolved_rows} resolved days.",
                evidence.reliability.adjusted_accuracy)
    if evidence.observation.is_same_day and evidence.observation.alignment_score >= 0.75:
        log.add("OBSERVATIONS_ALIGNED", "positive",
                "Live station observations are tracking the forecast.",
                evidence.observation.alignment_score)
    if evidence.observation.is_same_day and evidence.observation.anomaly_f is not None:
        anomaly = evidence.observation.anomaly_f
        if anomaly > 2.0:
            log.add("OBSERVATIONS_RUNNING_HOT", "high",
                    f"Station is {anomaly:.1f}F above the forecast's own expectation for this hour.", anomaly)
        elif anomaly < -2.0:
            log.add("OBSERVATIONS_RUNNING_COLD", "high",
                    f"Station is {abs(anomaly):.1f}F below the forecast's own expectation for this hour.", anomaly)
    if evidence.ensemble.outliers:
        log.add("MODEL_DISAGREEMENT", "medium",
                f"Outlier guidance source(s): {', '.join(evidence.ensemble.outliers)}.")
    if evidence.reliability.status == "building":
        log.add("INSUFFICIENT_HISTORY", "medium",
                f"{evidence.reliability.station} has only {evidence.reliability.resolved_rows} resolved days; "
                "its own record cannot yet raise this score.",
                evidence.reliability.resolved_rows)


def _calibrated(bucket: BucketProbability, calibrator: ProbabilityCalibrator) -> BucketProbability:
    return dataclasses_replace(bucket, probability=calibrator.apply(bucket.probability))


def decide(evidence: BetEvidence, config: BetFilterConfig | None = None) -> BetDecision:
    """Produce the final recommendation for one market."""
    config = config or DEFAULT_CONFIG
    log = ReasonLog()
    buckets = candidate_buckets(
        evidence.distribution,
        bucket_width_f=evidence.bucket_width_f,
        rounding=config.settlement_rounding,
        anchor_f=evidence.market_bucket[0] if evidence.market_bucket else None,
    )
    if evidence.market_bucket is not None:
        lower, upper = evidence.market_bucket
        selected = evidence.distribution.bucket_probability(lower, upper, config.settlement_rounding)
    else:
        selected = buckets[0] if buckets else None
    # Every downstream gate is a probability threshold, so the correction has
    # to land before the gates run -- not as a display-only adjustment.
    if config.apply_probability_calibration and evidence.calibrator.knots_x:
        buckets = sorted(
            (_calibrated(bucket, evidence.calibrator) for bucket in buckets),
            key=lambda bucket: bucket.probability,
            reverse=True,
        )
        if selected is not None:
            selected = _calibrated(selected, evidence.calibrator)
    gap = probability_gap(buckets)
    entropy = normalized_entropy(buckets)

    terminal = _run_hard_gates(evidence, selected, gap, config, log)
    if terminal is not None:
        return BetDecision(
            station=evidence.station, target_date=evidence.target_date, tier=terminal,
            label=TIER_LABELS[terminal], recommended=False, quality_score=0.0,
            selected_bucket=selected, bucket_probability=None if selected is None else selected.probability,
            probability_gap=gap, distribution_entropy=entropy, components=[],
            reasons=log.to_list(), alternatives=buckets, mode=config.mode,
        )

    assert selected is not None  # every path that leaves it None returned above
    quality_score, components = score(evidence, selected, config)
    # The concentration component needs the whole distribution, which the
    # per-bucket scorer does not see.
    concentration = _clip01(1.0 - entropy)
    for index, component in enumerate(components):
        if component.name == "Weather uncertainty":
            points = 100.0 * component.weight * concentration
            quality_score += points - component.weighted_points
            components[index] = ScoreComponent(
                name=component.name, raw=concentration, weight=component.weight,
                weighted_points=points, maximum_points=component.maximum_points,
            )
    _log_supporting_evidence(evidence, selected, log)

    tiers = config.tiers
    if quality_score >= tiers.elite:
        tier: Tier = "ELITE"
    elif quality_score >= tiers.strong:
        tier = "STRONG"
    elif quality_score >= tiers.playable:
        tier = "PLAYABLE"
    elif quality_score >= tiers.marginal:
        tier = "MARGINAL"
    else:
        tier = "PASS"
    if tier in {"PASS", "MARGINAL", "PLAYABLE"}:
        log.add("BELOW_QUALITY_THRESHOLD",
                "critical" if tier == "PASS" else "high",
                f"Bet quality {quality_score:.0f}/100 is below the {config.mode} mode's "
                f"{tiers.strong:.0f} recommendation threshold.",
                quality_score, tiers.strong)
    return BetDecision(
        station=evidence.station, target_date=evidence.target_date, tier=tier,
        label=TIER_LABELS[tier], recommended=tier in RECOMMENDED_TIERS,
        quality_score=quality_score, selected_bucket=selected,
        bucket_probability=selected.probability, probability_gap=gap,
        distribution_entropy=entropy, components=components,
        reasons=log.to_list(), alternatives=buckets, mode=config.mode,
    )


def summarize(decisions: list[BetDecision]) -> dict[str, object]:
    """Board-level counts for the dashboard header."""
    counts = {tier: 0 for tier in TIER_LABELS}
    for decision in decisions:
        counts[decision.tier] += 1
    recommended = [item for item in decisions if item.recommended]
    passed = [item for item in decisions if item.tier in {"PASS", "DATA_INSUFFICIENT"}]
    evaluated = len(decisions)
    return {
        "evaluated": evaluated,
        "counts": counts,
        "recommended": len(recommended),
        "coverageRate": round(len(recommended) / evaluated, 4) if evaluated else 0.0,
        "averageRecommendedProbability": (
            round(float(np.mean([item.bucket_probability for item in recommended])), 4)
            if recommended else None
        ),
        "averagePassProbability": (
            round(float(np.mean([item.bucket_probability for item in passed
                                 if item.bucket_probability is not None])), 4)
            if any(item.bucket_probability is not None for item in passed) else None
        ),
    }
