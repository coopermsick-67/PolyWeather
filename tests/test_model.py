from datetime import date, timedelta

import numpy as np
import pandas as pd

from polyweather.model import (
    CONFORMAL_NOMINAL_COVERAGE,
    CONFORMAL_QUANTILE,
    CONFORMAL_QUANTILE_MAX,
    CONFORMAL_SAFETY_MARGIN,
    AdaptiveResidualForecaster,
    BlendedResidualForecaster,
    ResidualForecaster,
    SeasonalClimatology,
    _select_conformal_quantile_level,
)
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


def test_select_conformal_quantile_level_widens_for_undercoverage():
    """A calibration block whose later half has visibly larger residuals than
    its earlier half (the shape a real undercovering period has) must raise
    the applied level above the base quantile, and the safety margin must
    never choose a *narrower* interval than omitting it would."""
    rng = np.random.default_rng(11)
    early = np.abs(rng.normal(2.0, 0.5, size=200))
    later = np.abs(rng.normal(4.0, 1.0, size=200))
    ordered = np.concatenate([early, later])
    level_no_margin = _select_conformal_quantile_level(
        ordered, CONFORMAL_QUANTILE, CONFORMAL_NOMINAL_COVERAGE, safety_margin=0.0
    )
    level_with_margin = _select_conformal_quantile_level(
        ordered, CONFORMAL_QUANTILE, CONFORMAL_NOMINAL_COVERAGE, safety_margin=CONFORMAL_SAFETY_MARGIN
    )
    assert level_no_margin > CONFORMAL_QUANTILE
    assert level_with_margin >= level_no_margin
    assert level_with_margin <= CONFORMAL_QUANTILE_MAX


def test_select_conformal_quantile_level_does_not_max_out_on_stable_residuals():
    """Residuals with no real scale change between calibration halves should
    not be pushed to the max quantile purely from split-half sampling noise."""
    rng = np.random.default_rng(3)
    ordered = np.abs(rng.normal(2.0, 0.5, size=400))
    level = _select_conformal_quantile_level(ordered, CONFORMAL_QUANTILE, CONFORMAL_NOMINAL_COVERAGE)
    assert level < CONFORMAL_QUANTILE_MAX


def _drifting_training_data(n_days: int = 500, seed: int = 7, drift_strength: float = 0.6) -> pd.DataFrame:
    """Ten stations with heavy-tailed (Student-t) residuals whose scale grows
    over the archive -- the realistic shape of a widening station roster or a
    changing NBM era, not a stationary noise process."""
    rng = np.random.default_rng(seed)
    stations = ["KNYC", "KMIA", "KMDW", "KLAX", "KSFO", "KDEN", "KPHX", "KSEA", "KBOS", "KATL"]
    station_scale = {s: rng.uniform(1.0, 3.0) for s in stations}
    rows = []
    start = date(2024, 10, 8)
    for offset in range(n_days):
        current = start + timedelta(days=offset)
        seasonal = 15 * np.sin(2 * np.pi * offset / 365)
        drift = 1.0 + drift_strength * (offset / n_days)
        for station in stations:
            nbm = 60 + seasonal + rng.normal(0, 0.3)
            scale = station_scale[station] * drift
            residual = rng.standard_t(df=4) * scale
            rows.append(
                {
                    "station": station,
                    "target_date": current,
                    "tmax_f": nbm + residual,
                    "nbm_baseline_f": nbm,
                    "ncep_nbm_conus__tmax_f": nbm,
                    "ncep_hrrr_conus__tmax_f": nbm + rng.normal(0, 0.4),
                    "dayofyear_sin": np.sin(2 * np.pi * offset / 365),
                    "dayofyear_cos": np.cos(2 * np.pi * offset / 365),
                }
            )
    return pd.DataFrame(rows)


def test_conformal_interval_survives_realistic_drift_into_the_future():
    """Regression test for a real undercoverage failure mode found during
    review: fitting on data with a slowly widening residual scale, then
    scoring on a genuinely held-out future window (never used for fitting or
    calibration), used to land coverage several points under the nominal 75%
    target -- the self-correction's nested split only ever looks one
    half-calibration-window ahead and could not see the trend continuing
    beyond it. Before CONFORMAL_SAFETY_MARGIN this exact scenario measured
    ~0.72 future coverage; assert it now clears the nominal target instead of
    silently regressing back under it."""
    frame = _drifting_training_data()
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    split_day = pd.Timestamp(date(2024, 10, 8)) + pd.Timedelta(days=420)
    train = frame[frame["target_date"] < split_day]
    future = frame[frame["target_date"] >= split_day]
    model = ResidualForecaster.fit(train, "ridge", calibration_days=60)
    prediction = model.predict(future)
    actual = future["tmax_f"].to_numpy()
    covered = (actual >= prediction["p10_f"].to_numpy()) & (actual <= prediction["p90_f"].to_numpy())
    assert covered.mean() >= CONFORMAL_NOMINAL_COVERAGE
    assert model.conformal_quantile_level >= CONFORMAL_QUANTILE


def test_conformal_interval_stays_reasonably_sharp_without_drift():
    """The safety margin trades some sharpness for coverage safety, but a
    stable (non-drifting) period should not be pushed anywhere near the
    0.95 ceiling -- it should still land in a plausible high-70s/low-80s band."""
    frame = _drifting_training_data(seed=11, drift_strength=0.0)
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    split_day = pd.Timestamp(date(2024, 10, 8)) + pd.Timedelta(days=420)
    train = frame[frame["target_date"] < split_day]
    future = frame[frame["target_date"] >= split_day]
    model = ResidualForecaster.fit(train, "ridge", calibration_days=60)
    prediction = model.predict(future)
    actual = future["tmax_f"].to_numpy()
    covered = (actual >= prediction["p10_f"].to_numpy()) & (actual <= prediction["p90_f"].to_numpy())
    assert CONFORMAL_NOMINAL_COVERAGE <= covered.mean() <= 0.90
    assert model.conformal_quantile_level <= 0.90


def test_adaptive_forecaster_selection_does_not_reuse_final_calibration_window():
    """Regression test: AdaptiveResidualForecaster used to pick Ridge vs XGB
    per station by scoring each candidate on the exact same trailing window
    later reused as that candidate's own offset-calibration block, so a
    family could "win" selection merely because its offset was fit to fit
    that window well (optimistic selection bias), not because it actually
    generalizes. The selector must now use its own held-out window, strictly
    separate from the calibration block: total data must exceed
    calibration + selection + a real training remainder."""
    frame = _synthetic_training_data(days=220)
    model = AdaptiveResidualForecaster.fit(frame, calibration_days=30, selection_days=30)
    # Every station must get an explicit choice; none silently missing.
    assert set(frame["station"]) == set(model.station_models)
    # The final members are fit on ALL usable history (not truncated to
    # exclude the selection window), matching BlendedResidualForecaster.
    assert model.xgb.train_rows + model.xgb.calibration_rows == len(frame)
    prediction = model.predict(frame.tail(5))
    assert (prediction["p10_f"] <= prediction["prediction_f"]).all()
    assert (prediction["prediction_f"] <= prediction["p90_f"]).all()
