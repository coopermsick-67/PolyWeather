from datetime import date, datetime, timezone

import pytest

from polyweather.markets import MarketType, QualityStatus, import_markets, normal_bracket_probabilities, parse_manual_market, quality_gate
from polyweather.nws import SourceSnapshot, daily_observation_extremes, normalized_observations
from polyweather.stations import require_station


def test_manual_market_maps_known_alias_and_parses_bracket():
    market = parse_manual_market({"marketId": "m1", "location": "Chicago", "targetDate": "2026-09-01", "title": "Chicago daily high 80-81F"})
    assert market.station_id == "KMDW"
    assert market.market_type == MarketType.TEMPERATURE_BRACKET
    assert (market.lower_f, market.upper_f) == (80.0, 81.0)


def test_unknown_market_location_requires_manual_review():
    with pytest.raises(ValueError, match="Manual-review"):
        parse_manual_market({"location": "Springfield", "targetDate": "2026-09-01", "title": "daily high"})


def test_quality_gate_refuses_stale_conflicting_or_wide_predictions():
    assert quality_gate(station_known=True, rules_available=True, freshness=.2, agreement=.9, interval_width_f=2) == QualityStatus.STALE_DATA
    assert quality_gate(station_known=True, rules_available=True, freshness=.9, agreement=.2, interval_width_f=2) == QualityStatus.SOURCE_CONFLICT
    assert quality_gate(station_known=True, rules_available=True, freshness=.9, agreement=.9, interval_width_f=10) == QualityStatus.NO_BET


def test_bracket_probabilities_are_bounded_and_centered():
    values = normal_bracket_probabilities(80, 2, [(78, 79), (80, 81)])
    assert all(0 <= value <= 1 for value in values.values())
    assert values["80-81"] > values["78-79"]


def test_observation_aggregation_is_local_date_and_deduplicated_across_dst():
    station = require_station("KMDW")
    snapshot = SourceSnapshot("NWS", datetime.now(timezone.utc), None, False, {"features": [
        {"properties": {"timestamp": "2026-03-08T07:30:00+00:00", "temperature": {"value": 10}}},
        {"properties": {"timestamp": "2026-03-08T08:30:00+00:00", "temperature": {"value": 12}}},
        {"properties": {"timestamp": "2026-03-08T08:30:00+00:00", "temperature": {"value": 13}}},
    ]})
    rows = normalized_observations(snapshot, station)
    assert len(rows) == 2
    high, low = daily_observation_extremes(rows, station, date(2026, 3, 8))
    assert round(high, 1) == 55.4
    assert round(low, 1) == 50.0


def test_import_rejects_bad_records_without_dropping_valid_records(tmp_path):
    path = tmp_path / "markets.json"
    path.write_text('[{"location":"Miami","targetDate":"2026-09-01","title":"daily high"},{"location":"Unknown","targetDate":"2026-09-01"}]')
    accepted, rejected = import_markets(path)
    assert len(accepted) == 1
    assert len(rejected) == 1
