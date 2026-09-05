"""Threshold contracts are the only class where a >=85% win rate survived
held-out measurement, so the arithmetic behind them is worth pinning down."""

from __future__ import annotations

import numpy as np
import pytest

from polyweather.betfilter.distribution import (
    TemperatureDistribution,
    from_calibrated_interval,
)
from polyweather.betfilter.threshold import (
    contract_for_target,
    win_probability,
)


def _distribution() -> TemperatureDistribution:
    return from_calibrated_interval(94.5, 92.0, 97.0, 0.80)


def test_settlement_rounding_is_applied_to_the_threshold():
    """A '>=95F' contract is won by a true high of 94.5F, because settlement
    rounds before comparing. Comparing against the printed integer would
    misprice every threshold by half a degree."""
    distribution = _distribution()
    assert win_probability(distribution, 95, "gte") == pytest.approx(
        1.0 - distribution.cdf(94.5)
    )
    assert win_probability(distribution, 95, "lte") == pytest.approx(
        distribution.cdf(95.5)
    )


def test_opposing_sides_overlap_rather_than_summing_to_one():
    """An observed high that rounds to exactly 95 wins both '>=95' and
    '<=95'. Forcing these to be complements would quietly understate one."""
    distribution = _distribution()
    total = win_probability(distribution, 95, "gte") + win_probability(distribution, 95, "lte")
    assert total > 1.0


def test_a_requested_target_is_met_not_merely_approached():
    """Rounding the solved line inward would return a contract sitting just
    under the confidence the caller asked for."""
    distribution = _distribution()
    for target in (0.85, 0.90, 0.95, 0.99):
        for side in ("gte", "lte"):
            contract = contract_for_target(distribution, target, side)
            assert contract.win_probability >= target
            assert win_probability(distribution, contract.threshold_f, side) >= target


def test_further_lines_are_safer_and_the_margin_is_signed_consistently():
    distribution = _distribution()
    safe = contract_for_target(distribution, 0.95, "gte")
    loose = contract_for_target(distribution, 0.85, "gte")
    assert safe.threshold_f <= loose.threshold_f
    assert safe.margin_from_forecast_f >= loose.margin_from_forecast_f


def test_an_observed_high_floor_collapses_a_settled_side():
    """Once today's high has already passed a line, that contract is decided;
    reporting a prior probability for it would be reporting a stale belief."""
    distribution = from_calibrated_interval(94.5, 92.0, 97.0, 0.80)
    floored = TemperatureDistribution(
        expected_high_f=distribution.expected_high_f,
        sigma_f=distribution.sigma_f,
        sigma_lower_f=distribution.sigma_lower_f,
        sigma_upper_f=distribution.sigma_upper_f,
        method=distribution.method,
        sample_size=distribution.sample_size,
        observed_high_floor_f=96.0,
    )
    assert win_probability(floored, 93, "gte") == pytest.approx(1.0)
    assert win_probability(floored, 93, "lte") == pytest.approx(0.0)


def test_an_empirical_step_distribution_still_reaches_its_target():
    """Empirical residuals make the CDF a step function, so solving for the
    line can need more than one outward step."""
    rng = np.random.default_rng(20260905)
    residuals = rng.normal(0.0, 2.0, size=400)
    distribution = TemperatureDistribution(
        expected_high_f=94.5, sigma_f=2.0, sigma_lower_f=2.0, sigma_upper_f=2.0,
        method="empirical", sample_size=residuals.size, residuals=residuals,
    )
    contract = contract_for_target(distribution, 0.90, "gte")
    assert contract.win_probability >= 0.90


def test_an_unknown_side_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError):
        win_probability(_distribution(), 95, "over")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        contract_for_target(_distribution(), 0.9, "under")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        contract_for_target(_distribution(), 1.0, "gte")
