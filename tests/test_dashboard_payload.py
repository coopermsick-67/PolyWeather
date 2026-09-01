from datetime import date

from polyweather.dashboard_payload import _prune_stability_state, stabilize_display_high


def test_display_high_holds_minor_model_churn():
    assert stabilize_display_high(previous=86, candidate=87) == (86, "held_minor_change")
    assert stabilize_display_high(previous=86, candidate=85) == (86, "held_minor_change")


def test_display_high_allows_material_or_observed_change():
    assert stabilize_display_high(previous=86, candidate=88) == (88, "material_model_change")
    assert stabilize_display_high(previous=86, candidate=85, observed_high=87) == (87, "observed_high")


def test_stability_cache_discards_expired_and_malformed_entries():
    state = {
        "forecasts": {
            "model::2026-07-01::KNYC": {"display_high_f": 82},
            "2026-07-01::KMIA": {"display_high_f": 91},
            "model::2026-08-29::KLAX": {"display_high_f": 78},
            "not-a-date": {"display_high_f": 70},
        }
    }
    _prune_stability_state(state, date(2026, 8, 30))
    assert state["forecasts"] == {"model::2026-08-29::KLAX": {"display_high_f": 78}}
