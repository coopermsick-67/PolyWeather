import pandas as pd

from datetime import date

from polyweather.backtest import rolling_fold_ranges
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
