from datetime import date, timedelta

import numpy as np
import pandas as pd

from polyweather.model import AdaptiveResidualForecaster, BlendedResidualForecaster, ResidualForecaster, SeasonalClimatology
from polyweather.data import add_derived_forecast_features


def _synthetic_training_data(days: int = 140) -> pd.DataFrame:
    rows = []
    stations = ["KNYC", "KMIA", "KMDW", "KLAX", "KSFO"]
    start = date(2025, 1, 1)
    for offset in range(days):
        current = start + timedelta(days=offset)
        seasonal = 15 * np.sin(2 * np.pi * offset / 365)
        for station_index, station in enumerate(stations):
            nbm = 60 + station_index + seasonal
            residual = (station_index - 2) * 0.7 + 0.25 * np.cos(offset / 7)
            rows.append(
                {
                    "station": station,
                    "target_date": current,
                    "tmax_f": nbm + residual,
                    "nbm_baseline_f": nbm,
                    "ncep_nbm_conus__tmax_f": nbm,
                    "ncep_hrrr_conus__tmax_f": nbm + 0.5,
                    "dayofyear_sin": np.sin(2 * np.pi * offset / 365),
                    "dayofyear_cos": np.cos(2 * np.pi * offset / 365),
                }
            )
    return pd.DataFrame(rows)


def test_residual_forecaster_outputs_point_and_interval():
    frame = _synthetic_training_data()
    model = ResidualForecaster.fit(frame, "ridge", calibration_days=30)
    prediction = model.predict(frame.tail(5))
    assert prediction.columns.tolist() == [
        "prediction_f",
        "p10_f",
        "p50_f",
        "p90_f",
        "conformal_halfwidth_f",
        "calibration_offset_f",
    ]
    assert (prediction["p10_f"] <= prediction["prediction_f"]).all()
    assert (prediction["prediction_f"] <= prediction["p90_f"]).all()


def test_climatology_is_station_specific():
    frame = _synthetic_training_data()
    climatology = SeasonalClimatology.fit(frame)
    predictions = climatology.predict(frame.head(5))
    assert len(predictions) == 5
    assert len(set(predictions)) > 1


def test_adaptive_forecaster_routes_without_changing_interval_contract():
    frame = _synthetic_training_data()
    model = AdaptiveResidualForecaster.fit(frame, calibration_days=30)
    prediction = model.predict(frame.tail(5))
    assert set(model.station_models).issubset(set(frame["station"]))
    assert (prediction["p10_f"] <= prediction["prediction_f"]).all()
    assert (prediction["prediction_f"] <= prediction["p90_f"]).all()


def test_blended_forecaster_selects_only_valid_convex_weights():
    frame = _synthetic_training_data(days=220)
    model = BlendedResidualForecaster.fit(frame, calibration_days=30, selection_days=30)
    prediction = model.predict(frame.tail(5))
    assert set(model.station_ridge_weights).issubset(set(frame["station"]))
    assert all(0.0 <= weight <= 1.0 for weight in model.station_ridge_weights.values())
    assert (prediction["p10_f"] <= prediction["prediction_f"]).all()
    assert (prediction["prediction_f"] <= prediction["p90_f"]).all()


def test_derived_forecast_features_are_source_only_and_finite_when_inputs_exist():
    frame = add_derived_forecast_features(_synthetic_training_data())
    assert "model_agreement__tmax_f_spread" in frame
    assert frame["model_agreement__tmax_f_spread"].notna().all()
