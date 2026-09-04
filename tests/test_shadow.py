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


def test_shadow_record_captures_model_and_source_provenance(monkeypatch):
    class Model:
        kind = "xgb"
        train_rows = 123
        calibration_rows = 20
        numeric_columns = [
            "ncep_nbm_conus__tmax_f",
            "ncep_hrrr_conus__tmax_f",
            "ncep_gfs_seamless__tmax_f",
        ]

        def predict(self, _):
            return pd.DataFrame(
                [{
                    "prediction_f": 80.0,
                    "interval_lower_f": 77.0,
                    "interval_upper_f": 83.0,
                    "p10_f": 77.0,
                    "p50_f": 80.0,
                    "p90_f": 83.0,
                    "calibration_offset_f": 0.1,
                    "conformal_halfwidth_f": 3.0,
                }]
            )

    monkeypatch.setattr(
        "polyweather.shadow.fetch_live_forecast_features",
        lambda *_: {
            "nbm_baseline_f": 79.0,
            "ncep_nbm_conus__tmax_f": 79.0,
            "ncep_hrrr_conus__tmax_f": 80.0,
            "ncep_gfs_seamless__tmax_f": 81.0,
            "source_provider": "Open-Meteo Forecast API",
            "source_fetched_at_utc": "2026-08-01T12:00:00+00:00",
            "source_generationtime_ms": 1.2,
            "source_timezone": "America/New_York",
            "source_utc_offset_seconds": -14400,
            "forecast_lead_days": 1,
        },
    )
    target = (pd.Timestamp.now(tz="America/New_York") + pd.Timedelta(days=1)).date()
    record = create_shadow_records(Model(), [STATIONS["KNYC"]], target, model_artifact_sha256="abc")[0]
    assert record["guidance_complete"] is True
    assert record["model_identity"]["artifact_sha256"] == "abc"
    assert record["source_provenance"]["provider"] == "Open-Meteo Forecast API"
    assert record["forecast_lead_days"] == 1
    assert record["interval_lower_f"] == 77.0
