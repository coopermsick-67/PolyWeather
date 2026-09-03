import pandas as pd
from unittest.mock import patch

from datetime import date

from polyweather.backtest import _acceptance, rolling_fold_ranges
from polyweather.metrics import forecast_metrics, relative_mae_skill


def test_metrics_and_skill_use_identical_valid_rows():
    frame = pd.DataFrame(
        {
            "actual": [70.0, 80.0, 90.0, None],
            "model": [71.0, 79.0, 90.0, 95.0],
            "baseline": [73.0, 76.0, 94.0, 95.0],
        }
    )
    metrics = forecast_metrics(frame, "actual", "model")
    assert metrics["n"] == 3
    assert metrics["mae_f"] == 2 / 3
    assert relative_mae_skill(frame, "actual", "model", "baseline") > 0


def test_rolling_fold_ranges_are_contiguous_and_non_overlapping():
    ranges = rolling_fold_ranges(date(2025, 2, 1), date(2025, 5, 31), window_days=31)
    assert ranges == [
        (date(2025, 2, 1), date(2025, 3, 3)),
        (date(2025, 3, 4), date(2025, 4, 3)),
        (date(2025, 4, 4), date(2025, 5, 4)),
        (date(2025, 5, 5), date(2025, 5, 31)),
    ]


def test_release_gate_requires_every_expected_station_not_a_small_subset():
    by_station = pd.DataFrame(
        [{"model": "XGBoost residual", "station": "KATL", "mae_skill_vs_nbm": 0.20}]
    )

    def metrics(_, __, label):
        values = {
            "XGBoost residual": {"mae_f": 1.0, "mae_skill_vs_nbm": 0.20},
            "Station blend residual": {"mae_f": 1.1, "mae_skill_vs_nbm": 0.12},
            "Ridge residual": {"mae_f": 1.2, "mae_skill_vs_nbm": 0.10},
        }
        return values[label]

    with patch("polyweather.backtest._metric_row", side_effect=metrics):
        acceptance = _acceptance(pd.DataFrame(), by_station, expected_stations=["KATL", "KMIA"])

    assert acceptance["required_station_wins"] == 2
    assert acceptance["missing_expected_stations"] == ["KMIA"]
    assert acceptance["statistical_candidate_passed"] is False
