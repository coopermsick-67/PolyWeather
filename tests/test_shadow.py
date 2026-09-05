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
            "ncep_nbm_conus__availability": 1.0,
            "ncep_hrrr_conus__availability": 1.0,
            "ncep_gfs_seamless__availability": 1.0,
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


def test_shadow_run_records_outages_without_discarding_successful_stations(monkeypatch):
    from polyweather.data import SourceError
    from polyweather.shadow import create_shadow_run
    def capture(model, stations, target, **kwargs):
        if stations[0].icao == "KNYC":
            raise SourceError("upstream unavailable")
        return [{"station": stations[0].icao, "status": "forecast"}]
    monkeypatch.setattr("polyweather.shadow.create_shadow_records", capture)
    target = (pd.Timestamp.now(tz="America/New_York") + pd.Timedelta(days=1)).date()
    records = create_shadow_run(object(), [STATIONS["KNYC"], STATIONS["KMIA"]], target)
    assert [record["status"] for record in records] == ["unavailable", "forecast"]
    assert records[0]["forecast_f"] is None


def test_shadow_log_duplicate_check_is_serialized_across_processes(tmp_path):
    import subprocess
    import sys
    path = tmp_path / "concurrent.jsonl"
    script = '''
import sys
from polyweather.shadow import append_jsonl
try:
    append_jsonl([{"station":"KNYC", "target_date":"2026-08-01", "issue_time_contract":"fixed"}], sys.argv[1])
except ValueError:
    sys.exit(2)
'''
    writers = [subprocess.Popen([sys.executable, "-c", script, str(path)]) for _ in range(2)]
    assert sorted(writer.wait(timeout=30) for writer in writers) == [0, 2]
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("issued", ["2026-08-01T12:00:00+00:00", "2026-07-31T12:00:00", None])
def test_verification_rejects_nonprospective_or_untimed_records(tmp_path, issued):
    from polyweather.shadow import verify_shadow_log
    path = tmp_path / "late.jsonl"
    append_jsonl([{"station": "KNYC", "target_date": "2026-08-01", "issue_time_contract": "fixed",
                   "issue_time_utc": issued}], path)
    with pytest.raises(ValueError, match="timestamped before"):
        verify_shadow_log(path)


def test_verification_reports_outage_coverage_and_pending_labels(tmp_path, monkeypatch):
    from datetime import date
    from polyweather.shadow import verify_shadow_log
    path = tmp_path / "outcomes.jsonl"
    base = {"target_date": "2026-08-01", "issue_time_contract": "fixed", "issue_time_utc": "2026-07-31T12:00:00+00:00",
            "interval_lower_f": 77, "interval_upper_f": 83}
    append_jsonl([{**base, "station": "KNYC", "forecast_f": 80},
                  {**base, "station": "KMIA", "forecast_f": None, "status": "unavailable"},
                  {**base, "station": "KLAX", "forecast_f": 80}], path)
    monkeypatch.setattr("polyweather.shadow.fetch_ncei_daily_tmax", lambda *_: pd.DataFrame([
        {"station": station, "target_date": date(2026, 8, 1), "tmax_f": 81}
        for station in ("KNYC", "KMIA")]))
    _, metrics = verify_shadow_log(path)
    assert metrics["n"] == 1
    assert metrics["forecast_coverage"] == .5
    assert metrics["pending_labels"] == 1
    assert metrics["mature_opportunities"] == 3
