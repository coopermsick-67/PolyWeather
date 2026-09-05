from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from polyweather.data import (
    MODEL_SOURCES, SourceError, _summarize_hourly_forecast,
    fetch_ncei_daily_tmax, has_complete_core_guidance,
)
from polyweather.stations import STATIONS
from polyweather.model import _validate_training_contract, usable_training_rows


def core():
    return {key: value for model in MODEL_SOURCES
            for key, value in ((f"{model}__tmax_f", 80.0), (f"{model}__availability", 1.0))}


@pytest.mark.parametrize("value", [None, "", "bad", True, False, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["tmax_f", "availability"])
def test_missing_or_malformed_guidance_fails_closed(field, value):
    row = core()
    row[f"{MODEL_SOURCES[0]}__{field}"] = value
    assert not has_complete_core_guidance(row)


def test_missing_coverage_is_not_complete():
    row = core()
    del row[f"{MODEL_SOURCES[0]}__availability"]
    assert not has_complete_core_guidance(row)
    row = core()
    row[f"{MODEL_SOURCES[0]}__tmax_f"] = "80.2"
    assert has_complete_core_guidance(row)


def summarize(times, temperatures=None):
    return _summarize_hourly_forecast({"hourly": {
        "time": [str(t) for t in times],
        "temperature_2m_previous_day1": temperatures or [80.0] * len(times),
    }}, STATIONS["KNYC"], 1, (MODEL_SOURCES[0],))


def test_truncated_response_and_duplicate_hours_do_not_inflate_coverage():
    times = pd.date_range("2026-07-01", periods=4, freq="h")
    assert summarize(times).iloc[0]["ncep_nbm_conus__availability"] == pytest.approx(4 / 24)
    assert summarize([times[0]] * 24).iloc[0]["ncep_nbm_conus__availability"] == pytest.approx(1 / 24)


@pytest.mark.parametrize("day, hours", [("2026-03-08", 23), ("2026-11-01", 25)])
def test_coverage_uses_actual_dst_day_length(day, hours):
    start = pd.Timestamp(day, tz="America/New_York")
    end = pd.Timestamp(start.date() + timedelta(days=1), tz="America/New_York")
    times = pd.date_range(start, end, inclusive="left", freq="h").tz_localize(None)
    assert len(times) == hours
    assert summarize(times).iloc[0]["ncep_nbm_conus__availability"] == 1.0


def test_hourly_length_mismatch_is_a_source_error():
    with pytest.raises(SourceError, match="length mismatch"):
        summarize(pd.date_range("2026-07-01", periods=4, freq="h"), [80.0])


def test_label_quality_flags_and_nonfinite_labels_are_excluded(monkeypatch):
    station = STATIONS["KNYC"]
    records = [{"STATION": station.ghcn_id, "DATE": "2026-07-01", "TMAX": value,
                "TMAX_ATTRIBUTES": attributes}
               for value, attributes in [(80, ",,W,"), (99, ",X,W,"), (True, ",,W,"), ("inf", ",,W,")]]
    monkeypatch.setattr("polyweather.data._request_json", lambda *_: records)
    labels = fetch_ncei_daily_tmax([station], date(2026, 7, 1), date(2026, 7, 1))
    assert labels.tmax_f.tolist() == [80.0]


@pytest.mark.parametrize("lead", [None, np.nan, np.inf, 1.5, True, "bad", 2])
def test_training_horizon_rejects_fractional_missing_and_boolean_values(lead):
    with pytest.raises(ValueError, match="1-day"):
        _validate_training_contract(pd.DataFrame({"forecast_lead_days": [lead]}))


def test_old_training_tables_cannot_bypass_coverage_gate():
    good = {**core(), "tmax_f": 81.0, "nbm_baseline_f": 80.0}
    bad = {**good, "ncep_nbm_conus__availability": .167}
    frame = pd.DataFrame([good, bad, {**good, "tmax_f": np.inf}])
    assert usable_training_rows(frame).index.tolist() == [0]
