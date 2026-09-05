from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from polyweather.betfilter import (
    BetEvidence,
    BetFilterConfig,
    DataQuality,
    ForecastSnapshot,
    analyze_ensemble,
    analyze_observation,
    analyze_stability,
    assess_reliability,
    candidate_buckets,
    decide,
    financial_result,
    from_calibrated_interval,
    from_residual_history,
    normalized_entropy,
    probability_gap,
    settlement_window,
    shrink,
    summarize,
    wilson_interval,
)
from polyweather.betfilter.backtest import (
    brier_score,
    calibration_table,
    effectiveness,
    expected_calibration_error,
    sweep_thresholds,
)
from polyweather.betfilter import calibration
from polyweather.betfilter.observation import HeatingCurve
from polyweather.betfilter.reliability import reliability_component
from polyweather.betfilter.results import classify_loss, resolve, settled_in_bucket

NOW = datetime(2026, 9, 4, 15, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Settlement rounding and bucket geometry
# --------------------------------------------------------------------------

def test_settlement_window_accounts_for_rounding_to_nearest_degree():
    """A 94-95 bucket is won by any high in [93.5, 95.5) -- a full degree
    wider than the printed bounds, which is decisive for a 2F market."""
    assert settlement_window(94, 95, "nearest") == (93.5, 95.5)
    assert settlement_window(94, 95, "truncate") == (94.0, 96.0)
    assert settlement_window(94, 95, "exact") == (94.0, 95.0)


def test_settled_in_bucket_uses_the_settlement_window_not_the_printed_bounds():
    assert settled_in_bucket(93.6, 94, 95) is True
    assert settled_in_bucket(95.4, 94, 95) is True
    assert settled_in_bucket(95.6, 94, 95) is False
    assert settled_in_bucket(93.4, 94, 95) is False


def test_settlement_window_rejects_inverted_buckets():
    with pytest.raises(ValueError):
        settlement_window(95, 94)


# --------------------------------------------------------------------------
# Distribution
# --------------------------------------------------------------------------

def test_split_normal_reproduces_its_nominal_coverage_even_when_asymmetric():
    """The calibrated interval is genuinely asymmetric; the fitted
    distribution must put exactly the nominal mass between its endpoints or
    every bucket probability downstream inherits the error."""
    distribution = from_calibrated_interval(94.0, 91.5, 97.5, nominal_coverage=0.80)
    covered = distribution.cdf(97.5) - distribution.cdf(91.5)
    assert covered == pytest.approx(0.80, abs=1e-6)
    assert distribution.sigma_lower_f != pytest.approx(distribution.sigma_upper_f)


def test_bucket_probabilities_sum_to_approximately_one_over_a_wide_grid():
    distribution = from_calibrated_interval(94.0, 91.5, 96.5, nominal_coverage=0.80)
    buckets = candidate_buckets(distribution, bucket_width_f=2, span_f=20)
    assert sum(bucket.probability for bucket in buckets) == pytest.approx(1.0, abs=0.01)


def test_tighter_interval_concentrates_more_mass_in_the_central_bucket():
    tight = from_calibrated_interval(94.5, 93.8, 95.2, nominal_coverage=0.80)
    wide = from_calibrated_interval(94.5, 91.5, 97.5, nominal_coverage=0.80)
    tight_probability = tight.bucket_probability(94, 95).probability
    wide_probability = wide.bucket_probability(94, 95).probability
    assert tight_probability > wide_probability
    assert tight_probability > 0.85


def test_empirical_distribution_requires_a_real_sample():
    with pytest.raises(ValueError, match="at least"):
        from_residual_history(94.0, np.array([0.1, -0.2, 0.3]))


def test_empirical_distribution_inherits_skew_from_the_residual_history():
    """Real forecast errors are not symmetric: most days cluster slightly
    cool with an occasional big warm bust. A symmetric fit erases exactly
    that tail, which is the tail that loses a 2F bet."""
    rng = np.random.default_rng(5)
    skewed = np.concatenate([rng.normal(-0.5, 0.8, 400), rng.normal(3.0, 1.5, 120)])
    distribution = from_residual_history(94.0, skewed)
    assert distribution.method == "empirical"
    median = distribution.quantile(0.50)
    upper_tail = distribution.quantile(0.95) - median
    lower_tail = median - distribution.quantile(0.05)
    assert upper_tail > lower_tail * 1.5


def test_observed_high_floor_removes_impossible_mass_and_renormalizes():
    """Once today's high is 95F, the day cannot finish at 92F. The
    distribution must drop that mass rather than keep reporting it."""
    distribution = from_calibrated_interval(
        94.0, 91.5, 96.5, nominal_coverage=0.80, observed_high_floor_f=95.0
    )
    assert distribution.cdf(94.0) == 0.0
    assert distribution.probability_between(90.0, 94.9) == 0.0
    assert distribution.cdf(200.0) == pytest.approx(1.0, abs=1e-9)


def test_probability_gap_separates_a_real_favorite_from_a_coin_flip():
    concentrated = from_calibrated_interval(94.5, 93.9, 95.1, nominal_coverage=0.80)
    ambiguous = from_calibrated_interval(95.5, 92.5, 98.5, nominal_coverage=0.80)
    assert probability_gap(candidate_buckets(concentrated)) > 0.30
    assert probability_gap(candidate_buckets(ambiguous)) < 0.15


def test_normalized_entropy_is_low_when_one_bucket_dominates():
    concentrated = from_calibrated_interval(94.5, 94.1, 94.9, nominal_coverage=0.80)
    diffuse = from_calibrated_interval(94.5, 89.5, 99.5, nominal_coverage=0.80)
    assert normalized_entropy(candidate_buckets(concentrated)) < normalized_entropy(candidate_buckets(diffuse))


def test_boundary_safety_is_measured_in_sigma_not_raw_degrees():
    """0.5F of headroom is comfortable for a sharp station and meaningless
    for a noisy one, so the metric has to be scale-free."""
    sharp = from_calibrated_interval(94.8, 94.3, 95.3, nominal_coverage=0.80)
    noisy = from_calibrated_interval(94.8, 92.0, 97.6, nominal_coverage=0.80)
    sharp_bucket = sharp.bucket_probability(94, 95)
    noisy_bucket = noisy.bucket_probability(94, 95)
    assert sharp_bucket.minimum_edge_distance_f == pytest.approx(noisy_bucket.minimum_edge_distance_f)
    assert sharp_bucket.normalized_boundary_safety > noisy_bucket.normalized_boundary_safety


def test_candidate_buckets_respect_the_market_grid_offset():
    """A 94-95 board and a 93-94 board are different grids; scoring the
    wrong one answers a question nobody asked."""
    distribution = from_calibrated_interval(94.5, 92.5, 96.5, nominal_coverage=0.80)
    even = candidate_buckets(distribution, anchor_f=94)
    odd = candidate_buckets(distribution, anchor_f=93)
    assert any(bucket.lower_f == 94 for bucket in even)
    assert any(bucket.lower_f == 93 for bucket in odd)


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------

def test_ensemble_flags_a_single_divergent_source_without_letting_it_move_consensus():
    analysis = analyze_ensemble({"NWS": 94.0, "HRRR": 94.2, "RAP": 94.5, "GFS": 99.0})
    assert "GFS" in analysis.outliers
    assert analysis.median_f == pytest.approx(94.35, abs=0.01)
    assert analysis.spread_f == pytest.approx(5.0)
    assert analysis.agreement_score == 0.0


def test_ensemble_agreement_rewards_tight_clusters():
    tight = analyze_ensemble({"NWS": 94.0, "HRRR": 94.2, "RAP": 94.1, "GFS": 94.3})
    loose = analyze_ensemble({"NWS": 94.0, "HRRR": 95.6, "RAP": 93.1, "GFS": 96.2})
    assert tight.agreement_score > loose.agreement_score
    assert tight.agreement_score > 0.9


def test_ensemble_handles_a_total_absence_of_sources_without_inventing_agreement():
    analysis = analyze_ensemble({"NWS": None, "HRRR": None})
    assert analysis.source_count == 0
    assert analysis.agreement_score == 0.0


def test_ensemble_weights_apply_when_source_skill_is_known():
    equal = analyze_ensemble({"NWS": 94.0, "GFS": 98.0})
    weighted = analyze_ensemble({"NWS": 94.0, "GFS": 98.0}, weights={"NWS": 9.0, "GFS": 1.0})
    assert equal.weighted_mean_f == pytest.approx(96.0)
    assert weighted.weighted_mean_f == pytest.approx(94.4)


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------

def _snapshots(values: list[float], buckets: list[int] | None = None) -> list[ForecastSnapshot]:
    return [
        ForecastSnapshot(
            captured_at=NOW - timedelta(hours=len(values) - index - 1),
            predicted_high_f=value,
            bucket_lower_f=None if buckets is None else buckets[index],
        )
        for index, value in enumerate(values)
    ]


def test_a_single_snapshot_scores_zero_stability_not_neutral():
    """A brand-new forecast has no track record. Treating unknown as
    average is exactly how missing evidence becomes false confidence."""
    analysis = analyze_stability(_snapshots([94.0]), now=NOW)
    assert analysis.stability_score == 0.0
    assert analysis.trend_direction == "unknown"


def test_settled_forecast_scores_high_stability():
    analysis = analyze_stability(_snapshots([94.4, 94.5, 94.5, 94.6, 94.5]), now=NOW)
    assert analysis.stability_score > 0.85
    assert analysis.trend_direction == "flat"
    assert analysis.bucket_flips_12h == 0


def test_bucket_flips_are_penalized_harder_than_drift_within_a_bucket():
    drifting = analyze_stability(_snapshots([94.1, 94.3, 94.5, 94.7], [94, 94, 94, 94]), now=NOW)
    flipping = analyze_stability(_snapshots([94.1, 96.3, 94.5, 96.7], [94, 96, 94, 96]), now=NOW)
    assert flipping.bucket_flips_12h == 3
    assert flipping.stability_score < drifting.stability_score


def test_six_hour_revision_is_measured_over_the_window_not_the_whole_history():
    snapshots = [
        ForecastSnapshot(NOW - timedelta(hours=24), 99.0),
        ForecastSnapshot(NOW - timedelta(hours=5), 94.0),
        ForecastSnapshot(NOW - timedelta(hours=1), 94.4),
    ]
    analysis = analyze_stability(snapshots, now=NOW)
    assert analysis.change_6h_f is None  # no sufficiently close six-hour baseline
    assert analysis.change_24h_f == pytest.approx(-4.6)


def test_trend_consistency_separates_a_clean_move_from_churn():
    clean = analyze_stability(_snapshots([96.0, 95.5, 95.0, 94.5]), now=NOW)
    churn = analyze_stability(_snapshots([96.0, 94.0, 96.0, 94.5]), now=NOW)
    assert clean.trend_consistency > churn.trend_consistency


# --------------------------------------------------------------------------
# Reliability and small samples
# --------------------------------------------------------------------------

def test_three_wins_from_three_bets_is_not_a_hundred_percent_station():
    """The single most dangerous number in this system. Shrinkage must pull
    a perfect tiny sample back toward the prior."""
    record = assess_reliability(
        "KPHX", resolved_rows=3, exact_bucket_hits=3,
        prior_accuracy=0.55, prior_rows=40, min_rows_for_credit=25,
    )
    assert record.raw_accuracy == 1.0
    assert record.adjusted_accuracy < 0.65
    assert record.status == "building"


def test_a_thin_record_cannot_raise_the_score_above_neutral():
    thin = assess_reliability("KNEW", 5, 5, 0.55, 40, 25)
    assert reliability_component(thin, 25) <= 0.5


def test_a_genuinely_poor_record_still_lowers_the_score_regardless_of_sample():
    """Absence of evidence is not evidence of quality, but evidence of
    failure is evidence of failure."""
    poor = assess_reliability("KDEN", 120, 48, 0.55, 40, 25)
    assert poor.adjusted_accuracy < 0.50
    assert reliability_component(poor, 25) < 0.35
    assert poor.status in {"restricted", "caution"}


def test_large_sample_lets_a_station_earn_full_credit():
    strong = assess_reliability("KMIA", 400, 300, 0.55, 40, 25)
    assert strong.adjusted_accuracy > 0.70
    assert strong.status == "active"
    assert reliability_component(strong, 25) > 0.70


def test_shrinkage_moves_halfway_to_the_prior_at_prior_strength():
    assert shrink(20, 40, prior_rate=0.50, prior_rows=40) == pytest.approx(0.5 * (0.5 + 0.5))
    assert shrink(40, 40, prior_rate=0.50, prior_rows=40) == pytest.approx(0.75)


def test_wilson_interval_is_wide_for_small_samples_and_stays_in_bounds():
    narrow_low, narrow_high = wilson_interval(300, 400)
    wide_low, wide_high = wilson_interval(3, 4)
    assert (narrow_high - narrow_low) < (wide_high - wide_low)
    assert 0.0 <= wide_low <= wide_high <= 1.0


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------

def test_future_dated_market_gets_no_credit_from_todays_observations():
    """Borrowing today's readings to support tomorrow's high is the exact
    confusion this whole layer is meant to prevent."""
    alignment = analyze_observation(is_same_day=False, predicted_high_f=94.0, current_temperature_f=91.0)
    assert alignment.current_temperature_f is None
    assert alignment.alignment_score == 0.5
    assert alignment.bucket_already_impossible is False


def test_same_day_market_with_no_observation_scores_zero_not_neutral():
    alignment = analyze_observation(is_same_day=True, predicted_high_f=94.0)
    assert alignment.alignment_score == 0.0


def test_observed_high_beyond_the_bucket_makes_it_impossible():
    alignment = analyze_observation(
        is_same_day=True, predicted_high_f=94.0, observed_high_so_far_f=97.0, bucket_upper_f=95.5,
    )
    assert alignment.bucket_already_impossible is True
    assert alignment.alignment_score == 0.0


def test_alignment_scores_drop_when_the_station_runs_hot_against_its_own_curve():
    curve = HeatingCurve("KPHX", 9, {10: 8.0, 13: 4.0, 16: 0.5}, sample_size=90)
    aligned = analyze_observation(
        is_same_day=True, predicted_high_f=94.0, observed_high_so_far_f=90.0,
        local_hour=13, heating_curve=curve,
    )
    hot = analyze_observation(
        is_same_day=True, predicted_high_f=94.0, observed_high_so_far_f=93.5,
        local_hour=13, heating_curve=curve,
    )
    assert aligned.expected_by_now_f == pytest.approx(90.0)
    assert aligned.alignment_score > hot.alignment_score
    assert hot.anomaly_f == pytest.approx(3.5)


def test_heating_curve_interpolates_between_known_hours():
    curve = HeatingCurve("KPHX", 9, {10: 8.0, 14: 2.0}, sample_size=50)
    assert curve.remaining_at(12) == pytest.approx(5.0)
    assert curve.remaining_at(6) == pytest.approx(8.0)
    assert curve.remaining_at(20) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Decision engine
# --------------------------------------------------------------------------

def _evidence(
    *,
    interval=(93.9, 95.1),
    expected=94.5,
    sources=None,
    snapshots=None,
    resolved_rows=200,
    hits=130,
    is_calibrated=True,
    supported_horizon=True,
    completeness=0.98,
    verified=True,
    data_age=20.0,
    horizon_hours=8.0,
    same_day=True,
    observed_high=None,
    market_bucket=(94, 95),
) -> BetEvidence:
    distribution = from_calibrated_interval(expected, interval[0], interval[1], 0.80)
    ensemble = analyze_ensemble(sources or {"NWS": 94.4, "HRRR": 94.6, "RAP": 94.5, "GFS": 94.7})
    stability = analyze_stability(snapshots or _snapshots([94.4, 94.5, 94.5, 94.6, 94.5]), now=NOW)
    reliability = assess_reliability("KPHX", resolved_rows, hits, 0.55, 40, 25)
    observation = analyze_observation(
        is_same_day=same_day, predicted_high_f=expected, observed_high_so_far_f=observed_high,
        bucket_upper_f=95.5 if observed_high is not None else None,
    )
    return BetEvidence(
        station="KPHX", target_date="2026-09-04", is_same_day=same_day,
        distribution=distribution, ensemble=ensemble, stability=stability,
        reliability=reliability, observation=observation,
        data_quality=DataQuality(
            is_calibrated=is_calibrated, supported_horizon=supported_horizon,
            feature_completeness=completeness, source_count=len(sources or {}) or 4,
            settlement_station_verified=verified, data_age_minutes=data_age,
            forecast_horizon_hours=horizon_hours,
            source_run_age_verified=True,
            probability_calibration_verified=True,
        ),
        market_bucket=market_bucket,
    )


def test_a_genuinely_strong_setup_is_recommended():
    decision = decide(_evidence(observed_high=92.0))
    assert decision.tier in {"ELITE", "STRONG"}
    assert decision.recommended is True
    assert decision.bucket_probability > 0.70
    assert any(reason["severity"] == "positive" for reason in decision.reasons)


def test_low_probability_forces_a_pass_regardless_of_everything_else():
    """Every other signal here is excellent; the gate must still fire."""
    decision = decide(_evidence(interval=(90.0, 99.0), expected=94.5))
    assert decision.tier == "PASS"
    assert decision.recommended is False
    assert any(reason["code"] == "LOW_RANGE_PROBABILITY" for reason in decision.reasons)


def test_boundary_risk_forces_a_pass_even_when_probability_looks_acceptable():
    """A forecast pinned against a bucket edge is the classic 2F trap: the
    point estimate is inside the range and one degree of error is fatal."""
    decision = decide(_evidence(expected=95.45, interval=(94.6, 96.3), market_bucket=(94, 95)))
    codes = {reason["code"] for reason in decision.reasons}
    assert decision.recommended is False
    assert "HIGH_BOUNDARY_RISK" in codes or "LOW_RANGE_PROBABILITY" in codes


def test_ensemble_disagreement_forces_a_pass():
    decision = decide(_evidence(sources={"NWS": 94.0, "HRRR": 94.2, "RAP": 94.4, "GFS": 98.5}))
    assert decision.tier == "PASS"
    assert any(reason["code"] == "HIGH_ENSEMBLE_SPREAD" for reason in decision.reasons)


def test_late_forecast_movement_forces_a_pass():
    moving = [
        ForecastSnapshot(NOW - timedelta(hours=5), 92.0),
        ForecastSnapshot(NOW - timedelta(hours=3), 93.2),
        ForecastSnapshot(NOW - timedelta(hours=1), 94.5),
    ]
    decision = decide(_evidence(snapshots=moving))
    assert decision.tier == "PASS"
    assert any(reason["code"] == "FORECAST_UNSTABLE" for reason in decision.reasons)


def test_bucket_flips_force_a_pass():
    flipping = _snapshots([94.2, 96.3, 94.4, 96.6, 94.5], [94, 96, 94, 96, 94])
    decision = decide(_evidence(snapshots=flipping))
    assert decision.tier == "PASS"
    assert any(reason["code"] in {"BUCKET_FLIP_RISK", "FORECAST_UNSTABLE"} for reason in decision.reasons)


def test_already_impossible_bucket_is_never_recommended():
    decision = decide(_evidence(observed_high=97.0))
    assert decision.tier == "PASS"
    assert any(reason["code"] == "BUCKET_ALREADY_IMPOSSIBLE" for reason in decision.reasons)


def test_poor_station_history_forces_a_pass():
    decision = decide(_evidence(resolved_rows=200, hits=70))
    assert decision.tier == "PASS"
    assert any(reason["code"] == "LOW_STATION_RELIABILITY" for reason in decision.reasons)


# --- fail-safe: missing data must never become confidence -----------------

def test_missing_features_return_data_insufficient_not_pass():
    """'We cannot tell' and 'we looked and it is not good enough' are
    different answers; collapsing them hides which one happened."""
    decision = decide(_evidence(completeness=0.40))
    assert decision.tier == "DATA_INSUFFICIENT"
    assert decision.recommended is False
    assert any(reason["code"] == "LOW_FEATURE_COMPLETENESS" for reason in decision.reasons)


def test_uncalibrated_station_returns_data_insufficient():
    decision = decide(_evidence(is_calibrated=False))
    assert decision.tier == "DATA_INSUFFICIENT"


def test_unverified_settlement_station_returns_data_insufficient():
    decision = decide(_evidence(verified=False))
    assert decision.tier == "DATA_INSUFFICIENT"


def test_stale_data_returns_data_insufficient():
    decision = decide(_evidence(data_age=600.0))
    assert decision.tier == "DATA_INSUFFICIENT"
    assert any(reason["code"] == "STALE_DATA" for reason in decision.reasons)


def test_too_few_sources_returns_data_insufficient():
    decision = decide(_evidence(sources={"NWS": 94.5}))
    assert decision.tier == "DATA_INSUFFICIENT"
    assert any(reason["code"] == "INSUFFICIENT_DATA_SOURCES" for reason in decision.reasons)


def test_unsupported_horizon_returns_data_insufficient():
    decision = decide(_evidence(supported_horizon=False))
    assert decision.tier == "DATA_INSUFFICIENT"


# --- selectivity modes ----------------------------------------------------

def test_stricter_modes_only_move_thresholds_never_the_forecast():
    evidence = _evidence(expected=94.5, interval=(93.5, 95.5), observed_high=92.0)
    standard = decide(evidence, BetFilterConfig().for_mode("standard"))
    conservative = decide(evidence, BetFilterConfig().for_mode("conservative"))
    strict = decide(evidence, BetFilterConfig().for_mode("very_conservative"))
    assert standard.bucket_probability == conservative.bucket_probability == strict.bucket_probability
    tier_rank = {"ELITE": 5, "STRONG": 4, "PLAYABLE": 3, "MARGINAL": 2, "PASS": 1, "DATA_INSUFFICIENT": 0}
    assert tier_rank[strict.tier] <= tier_rank[conservative.tier] <= tier_rank[standard.tier]


def test_very_conservative_mode_rejects_a_setup_standard_mode_allows():
    """The measured evidence is identical in both modes -- only the bar
    moves. A gated decision reports score 0 because it was never scored,
    which is the honest representation of 'we stopped before scoring'."""
    evidence = _evidence(expected=94.5, interval=(93.2, 95.8), observed_high=92.0)
    standard = decide(evidence, BetFilterConfig().for_mode("standard"))
    strict = decide(evidence, BetFilterConfig().for_mode("very_conservative"))
    assert standard.bucket_probability == pytest.approx(strict.bucket_probability)
    assert not (strict.recommended and not standard.recommended)
    if strict.tier == "PASS" and standard.recommended:
        assert any(reason["severity"] == "critical" for reason in strict.reasons)


def test_scoring_weights_must_sum_to_one():
    from polyweather.betfilter.config import ScoringWeights

    with pytest.raises(ValueError, match="sum to 1.0"):
        BetFilterConfig(weights=ScoringWeights(range_probability=0.9))


def test_marginal_is_not_a_recommendation():
    """MARGINAL exists to be shown and skipped. If it ever counts as a bet
    the whole selectivity premise collapses."""
    from polyweather.betfilter import RECOMMENDED_TIERS

    assert "MARGINAL" not in RECOMMENDED_TIERS
    assert "PLAYABLE" not in RECOMMENDED_TIERS


def test_every_decision_explains_itself():
    for evidence in (_evidence(observed_high=92.0), _evidence(interval=(90.0, 99.0)), _evidence(completeness=0.1)):
        decision = decide(evidence)
        assert decision.reasons, f"{decision.tier} produced no reasons"
        assert all({"code", "severity", "message"} <= set(reason) for reason in decision.reasons)


def test_summarize_reports_board_level_selectivity():
    decisions = [
        decide(_evidence(observed_high=92.0)),
        decide(_evidence(interval=(90.0, 99.0))),
        decide(_evidence(completeness=0.2)),
    ]
    summary = summarize(decisions)
    assert summary["evaluated"] == 3
    assert 0.0 <= summary["coverageRate"] <= 1.0
    assert sum(summary["counts"].values()) == 3


# --------------------------------------------------------------------------
# Financial results -- the platform label is not the metric
# --------------------------------------------------------------------------

def test_financial_result_counts_only_money_gained_as_a_win():
    assert financial_result(5.0, 8.0) == "WIN"
    assert financial_result(5.0, 5.0) == "LOSS"
    assert financial_result(5.0, 4.99) == "LOSS"
    assert financial_result(5.0, 0.0) == "LOSS"


def test_a_platform_win_that_lost_money_is_recorded_as_a_loss():
    """PrizePicks shows $3.60 -> $2.56 as a win. It is a $1.04 loss and the
    strategy metrics must say so."""
    resolved = resolve(
        "KPHX", "2026-09-04", 94, 95, predicted_high_f=94.5, actual_high_f=94.0,
        buy_in=3.60, payout=2.56,
    )
    assert resolved.settled_in_bucket is True
    assert resolved.financial == "LOSS"
    assert resolved.net_profit == pytest.approx(-1.04)


def test_loss_classification_separates_boundary_misses_from_forecast_misses():
    boundary = classify_loss(95.4, 95.6, 94, 95)
    big_miss = classify_loss(94.5, 101.0, 94, 95)
    assert boundary == "BUCKET_BOUNDARY_MISS"
    assert big_miss == "UNDER_PREDICTED_TEMP"


def test_loss_classification_attributes_late_movement_and_disagreement():
    assert classify_loss(94.5, 99.0, 94, 95, forecast_change_6h_f=-2.2) == "FORECAST_CHANGED_LATE"
    assert classify_loss(94.5, 99.0, 94, 95, ensemble_spread_f=3.1) == "MODEL_DISAGREEMENT_IGNORED"


# --------------------------------------------------------------------------
# Filter effectiveness backtesting
# --------------------------------------------------------------------------

def _decision_log() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for index in range(400):
        probability = float(rng.uniform(0.40, 0.90))
        quality = float(np.clip(probability * 110 + rng.normal(0, 6), 0, 100))
        # Outcomes generated from the stated probability, so a working
        # filter must show a real lift and a broken one must not.
        rows.append({
            "station": f"K{index % 5}",
            "target_date": f"2026-08-{(index % 28) + 1:02d}",
            "bucket_probability": probability,
            "quality_score": quality,
            "settled_in_bucket": bool(rng.random() < probability),
            "primary_reason": None if probability > 0.6 else "LOW_RANGE_PROBABILITY",
        })
    return pd.DataFrame(rows)


def test_effectiveness_shows_a_real_lift_over_betting_every_favorite():
    result = effectiveness(_decision_log(), minimum_probability=0.70, minimum_quality_score=75.0)
    assert result.recommended_win_rate > result.unfiltered_win_rate
    assert result.lift_percentage_points > 0
    assert result.coverage_rate < 1.0
    assert result.rejected_would_have_lost > 0


def test_effectiveness_reports_rejected_markets_that_would_have_won():
    """A filter that rejects winners at the same rate it rejects losers is
    destroying opportunity, not risk, and this is where that shows up."""
    result = effectiveness(_decision_log(), minimum_probability=0.70, minimum_quality_score=75.0)
    assert result.rejected == result.rejected_would_have_won + result.rejected_would_have_lost
    assert 0.0 <= result.avoided_loss_rate <= 1.0


def test_threshold_sweep_flags_thresholds_with_too_few_bets():
    table = sweep_thresholds(_decision_log())
    assert not table.empty
    assert set(table.columns) >= {"minimum_probability", "win_rate", "coverage", "sufficient_sample"}
    strictest = table.loc[table["minimum_probability"] == 0.80]
    assert (strictest["coverage"] <= table["coverage"].max()).all()


def test_win_rate_rises_monotonically_enough_with_the_probability_threshold():
    table = sweep_thresholds(_decision_log(), quality_grid=(0.0,))
    usable = table.loc[table["sufficient_sample"]].sort_values("minimum_probability")
    assert usable["win_rate"].iloc[-1] > usable["win_rate"].iloc[0]


def test_calibration_table_detects_overconfident_probabilities():
    frame = _decision_log()
    # Force systematic overconfidence: outcomes far worse than claimed.
    frame["settled_in_bucket"] = frame["bucket_probability"] > 0.95
    error = expected_calibration_error(frame)
    assert error > 0.25
    assert brier_score(frame) > 0.25


def test_calibration_table_reports_bins_with_counts():
    table = calibration_table(_decision_log())
    assert not table.empty
    assert (table["n"] > 0).all()
    assert set(table.columns) >= {"bin_lower", "bin_upper", "n", "mean_predicted", "observed_rate", "gap"}


def test_backtest_rejects_a_log_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        effectiveness(pd.DataFrame({"station": ["KPHX"]}), 0.6, 75.0)


# --------------------------------------------------------------------------
# Probability calibration
# --------------------------------------------------------------------------

def test_calibration_corrects_systematic_overconfidence():
    """Measured on real held-out history, this system's raw bucket
    probabilities run several points hot. Every gate is a probability
    threshold, so an uncorrected probability silently moves every gate."""
    rng = np.random.default_rng(9)
    stated = rng.uniform(0.3, 0.85, 3000)
    # Outcomes occur at 70% of the stated rate: deliberate overconfidence.
    settled = (rng.random(3000) < stated * 0.7).astype(float)
    fitted = calibration.fit(stated, settled, fitted_through="2026-08-01")
    assert fitted.rows == 3000
    for probability in (0.50, 0.60, 0.70):
        assert fitted.apply(probability) < probability


def test_calibration_is_monotone_so_gate_ordering_survives():
    rng = np.random.default_rng(2)
    stated = rng.uniform(0.25, 0.9, 2000)
    settled = (rng.random(2000) < stated * 0.8).astype(float)
    fitted = calibration.fit(stated, settled)
    values = [fitted.apply(p) for p in np.linspace(0.25, 0.9, 40)]
    assert all(later >= earlier - 1e-9 for earlier, later in zip(values, values[1:], strict=False))


def test_calibration_refuses_to_fit_on_too_little_history():
    """Isotonic regression will happily carve a step function out of noise;
    below the minimum the honest answer is to leave probabilities alone."""
    rng = np.random.default_rng(1)
    fitted = calibration.fit(rng.uniform(0.3, 0.8, 50), rng.integers(0, 2, 50).astype(float))
    assert fitted.rows == 0
    assert fitted.apply(0.62) == 0.62


def test_calibration_does_not_manufacture_certainty_in_a_sparse_tail():
    """The top of the probability range is always the thinnest part of the
    sample. Unshrunk isotonic maps a handful of lucky high-stated markets
    onto a claimed certainty of 1.0 -- the one direction this correction
    must never move."""
    rng = np.random.default_rng(6)
    bulk = rng.uniform(0.30, 0.60, 2000)
    tail = np.full(12, 0.82)
    stated = np.concatenate([bulk, tail])
    settled = np.concatenate([
        (rng.random(2000) < bulk * 0.8).astype(float),
        np.ones(12),  # every sparse-tail market happened to hit
    ])
    fitted = calibration.fit(stated, settled)
    assert fitted.apply(0.82) < 1.0


def test_calibrated_probability_is_used_by_the_gates_not_just_displayed():
    """A correction applied only for display would leave every threshold
    operating on the uncorrected number."""
    stated = np.linspace(0.30, 0.995, 2000)
    settled = (np.random.default_rng(3).random(2000) < stated * 0.6).astype(float)
    fitted = calibration.fit(stated, settled)
    evidence = _evidence(observed_high=92.0)
    raw = decide(evidence)
    calibrated = decide(
        BetEvidence(**{**evidence.__dict__, "calibrator": fitted}),
        BetFilterConfig().for_mode("conservative"),
    )
    assert calibrated.bucket_probability < raw.bucket_probability
    tier_rank = {"ELITE": 5, "STRONG": 4, "PLAYABLE": 3, "MARGINAL": 2, "PASS": 1, "DATA_INSUFFICIENT": 0}
    assert tier_rank[calibrated.tier] <= tier_rank[raw.tier]


def test_calibration_round_trips_through_disk(tmp_path):
    rng = np.random.default_rng(7)
    stated = rng.uniform(0.3, 0.85, 1500)
    fitted = calibration.fit(stated, (rng.random(1500) < stated * 0.75).astype(float))
    path = fitted.save(tmp_path / "calibrator.json")
    restored = calibration.ProbabilityCalibrator.load(path)
    assert restored.rows == fitted.rows
    assert restored.apply(0.65) == pytest.approx(fitted.apply(0.65))


def test_calibration_never_raises_confidence_beyond_the_range_it_was_fitted_on():
    """np.interp clamps outside the fitted range. Left unguarded, a market
    more confident than anything the calibrator ever saw picks up the last
    knot's value -- silently raising confidence in exactly the region with
    no evidence behind it."""
    rng = np.random.default_rng(12)
    stated = rng.uniform(0.30, 0.60, 2000)
    # Outcomes beat the stated rate, so the fit's top knot sits above 0.60.
    settled = (rng.random(2000) < np.minimum(stated * 1.5, 1.0)).astype(float)
    fitted = calibration.fit(stated, settled)
    assert fitted.apply(0.55) >= 0.55  # inside the range the fit may raise
    assert fitted.apply(0.95) <= 0.95  # outside it, never
    assert fitted.apply(0.10) <= 0.10


def test_a_forecast_with_no_revision_history_cannot_satisfy_the_stability_gate():
    """Every station-day's first refresh has a single snapshot, so 6-hour
    movement is unmeasurable. Folding that into ``or 0.0`` scored it as
    perfectly steady and let the least-evidenced forecast of the day pass
    the gate outright -- unknown must not read as calm."""
    decision = decide(_evidence(snapshots=_snapshots([94.5]), observed_high=92.0))
    assert decision.tier == "DATA_INSUFFICIENT"
    assert decision.recommended is False
    assert any(reason["code"] == "FORECAST_STABILITY_UNKNOWN" for reason in decision.reasons)


def test_a_measurable_revision_trail_still_passes_the_stability_gate():
    """The guard above must reject only unmeasurable movement, not tighten
    the threshold for forecasts that do carry a trail."""
    decision = decide(_evidence(snapshots=_snapshots([94.4, 94.5, 94.5]), observed_high=92.0))
    assert decision.tier != "DATA_INSUFFICIENT"
    assert not any(reason["code"] == "FORECAST_STABILITY_UNKNOWN" for reason in decision.reasons)


@pytest.mark.parametrize("age", [None, float("nan"), float("inf"), -1])
def test_unknown_or_invalid_run_age_cannot_satisfy_freshness(age):
    decision = decide(_evidence(data_age=age))
    assert decision.tier == "DATA_INSUFFICIENT"
    assert decision.reasons[0]["code"] == "SOURCE_RUN_AGE_UNVERIFIED"


@pytest.mark.parametrize("field, reason", [
    ("source_run_age_verified", "SOURCE_RUN_AGE_UNVERIFIED"),
    ("probability_calibration_verified", "PROBABILITY_CALIBRATION_UNVERIFIED"),
])
def test_unverified_evidence_blocks_an_otherwise_strong_recommendation(field, reason):
    from dataclasses import replace
    evidence = _evidence()
    decision = decide(replace(evidence, data_quality=replace(evidence.data_quality, **{field: False})))
    assert decision.tier == "DATA_INSUFFICIENT"
    assert decision.reasons[0]["code"] == reason


def test_nonfinite_completeness_cannot_satisfy_quality_gate():
    assert decide(_evidence(completeness=float("nan"))).tier == "DATA_INSUFFICIENT"


def test_empirical_bucket_mass_uses_same_half_open_window_as_settlement():
    distribution = from_residual_history(0.0, np.tile([93.5, 95.5], 60))
    assert distribution.bucket_probability(94, 95).probability == .5
    assert distribution.bucket_probability(96, 97).probability == .5
    assert distribution.bucket_probability(92, 93).probability == 0


def test_floor_keeps_empirical_mass_at_the_observed_high_and_conditions_quantiles():
    distribution = from_residual_history(0, np.tile([91.5, 93.5, 95.5], 40), observed_high_floor_f=93.5)
    assert distribution.bucket_probability(94, 95).probability == pytest.approx(.5)
    assert distribution.quantile(.1) >= 93.5
    assert distribution.cdf(93.5) == pytest.approx(.5)


def test_zero_tail_floor_retains_a_point_mass_in_its_bucket():
    distribution = from_residual_history(0, np.zeros(120), observed_high_floor_f=93.5)
    assert distribution.bucket_probability(94, 95).probability == 1
    assert distribution.quantile(.5) == 93.5


def test_bucket_grid_stays_anchored_when_span_is_not_divisible_by_width():
    distribution = from_calibrated_interval(81, 78, 84, .8)
    buckets = candidate_buckets(distribution, bucket_width_f=3, span_f=8, anchor_f=80)
    assert all((item.lower_f - 80) % 3 == 0 for item in buckets)


def test_refresh_padding_does_not_dilute_forecast_volatility():
    sparse = [ForecastSnapshot(NOW - timedelta(hours=6), 94), ForecastSnapshot(NOW, 96)]
    padded = [sparse[0], *[ForecastSnapshot(NOW - timedelta(minutes=m), 94) for m in range(359, 0, -1)], sparse[1]]
    assert analyze_stability(sparse, NOW).stability_score == pytest.approx(analyze_stability(padded, NOW).stability_score)


def test_future_snapshot_cannot_create_stability_evidence():
    snapshots = [ForecastSnapshot(NOW, 94), ForecastSnapshot(NOW + timedelta(hours=6), 94)]
    result = analyze_stability(snapshots, NOW)
    assert result.snapshot_count == 1
    assert result.stability_score == 0


def test_exact_upper_settlement_edge_is_already_impossible():
    alignment = analyze_observation(True, 94.5, observed_high_so_far_f=95.5, bucket_upper_f=95.5)
    assert alignment.bucket_already_impossible
