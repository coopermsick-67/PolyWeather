import pytest

from polyweather.stations import STATIONS, require_station


def test_all_requested_stations_have_distinct_ghcn_ids_and_timezones():
    assert len(STATIONS) == 20
    assert len({station.ghcn_id for station in STATIONS.values()}) == 20
    assert require_station("katl").timezone == "America/New_York"
    assert require_station("klax").timezone == "America/Los_Angeles"


def test_important_city_mappings_use_the_settlement_airport_not_a_nearby_airport():
    assert require_station("chicago").icao == "KMDW"
    assert require_station("dallas").icao == "KDFW"
    assert require_station("houston").icao == "KHOU"
    assert "not" in require_station("chicago").display_note.casefold()


def test_unknown_station_is_actionable():
    with pytest.raises(ValueError, match="KNYC"):
        require_station("KXYZ")


def test_every_registered_station_has_known_market_types():
    # dashboard_payload.py derives quality_gate's rules_available from
    # bool(station.market_types); every registered station must have at
    # least one configured market type or its data-quality badge would
    # incorrectly report UNKNOWN_SETTLEMENT_RULE.
    for station in STATIONS.values():
        assert station.market_types, f"{station.icao} has no configured market types"
