import pandas as pd
import pytest

from polyweather.shadow import append_jsonl, create_shadow_records
from polyweather.stations import STATIONS


def test_append_shadow_log_rejects_duplicate_station_date_contract(tmp_path):
    path = tmp_path / "shadow.jsonl"
    record = {
        "station": "KNYC",
        "target_date": "2026-08-14",
        "issue_time_contract": "latest live multi-model forecast at request time",
    }
    append_jsonl([record], path)
    with pytest.raises(ValueError, match="duplicate prospective snapshot"):
        append_jsonl([record], path)
    assert len([line for line in path.read_text(encoding="utf-8").splitlines() if line]) == 1


def test_shadow_records_reject_current_local_target_before_fetching_sources():
    local_today = pd.Timestamp.now(tz="America/New_York").date()
    with pytest.raises(ValueError, match="prospective logging requires"):
        create_shadow_records(object(), [STATIONS["KNYC"]], local_today)
