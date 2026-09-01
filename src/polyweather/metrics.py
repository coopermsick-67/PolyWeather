"""Forecast metrics, including strict model-versus-baseline skill."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return frame.loc[np.isfinite(values).all(axis=1)].copy()


def forecast_metrics(frame: pd.DataFrame, actual: str, prediction: str) -> dict[str, float]:
    """Return deterministic error metrics for one forecast column."""
    data = _finite(frame, actual, prediction)
    if data.empty:
        return {"n": 0, "mae_f": np.nan, "rmse_f": np.nan, "bias_f": np.nan,
                "median_ae_f": np.nan, "p90_ae_f": np.nan,
                "within_1f": np.nan, "within_2f": np.nan, "within_3f": np.nan}
    error = data[prediction].astype(float) - data[actual].astype(float)
    absolute_error = error.abs()
    return {
        "n": int(len(data)),
        "mae_f": float(absolute_error.mean()),
        "rmse_f": float(np.sqrt(np.mean(np.square(error)))),
        "bias_f": float(error.mean()),
        "median_ae_f": float(absolute_error.median()),
        "p90_ae_f": float(absolute_error.quantile(0.90)),
        "within_1f": float((absolute_error <= 1).mean()),
        "within_2f": float((absolute_error <= 2).mean()),
        "within_3f": float((absolute_error <= 3).mean()),
    }


def relative_mae_skill(frame: pd.DataFrame, actual: str, model: str, baseline: str) -> float:
    """Compute 1 - MAE(model) / MAE(baseline) on the identical rows."""
    data = _finite(frame, actual, model, baseline)
    if data.empty:
        return np.nan
    model_mae = (data[model] - data[actual]).abs().mean()
    baseline_mae = (data[baseline] - data[actual]).abs().mean()
    return float(np.nan if baseline_mae == 0 else 1 - model_mae / baseline_mae)


def interval_metrics(
    frame: pd.DataFrame,
    actual: str = "tmax_f",
    lower: str = "p10_f",
    upper: str = "p90_f",
) -> dict[str, float]:
    """Summarize empirical coverage and interval width."""
    data = _finite(frame, actual, lower, upper)
    if data.empty:
        return {"interval_n": 0, "coverage": np.nan, "mean_width_f": np.nan}
    contained = (data[actual] >= data[lower]) & (data[actual] <= data[upper])
    return {
        "interval_n": int(len(data)),
        "coverage": float(contained.mean()),
        "mean_width_f": float((data[upper] - data[lower]).mean()),
    }
