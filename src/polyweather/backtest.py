"""Rolling-origin backtests and acceptance tests for the residual forecaster."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import forecast_metrics, interval_metrics, relative_mae_skill
from .model import (
    BASELINE_COLUMN,
    CONFORMAL_NOMINAL_COVERAGE,
    TARGET_COLUMN,
    BlendedResidualForecaster,
    ResidualForecaster,
    SeasonalClimatology,
)
from .stations import STATIONS


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    overall: pd.DataFrame
    by_station: pd.DataFrame
    by_month: pd.DataFrame
    bootstrap: pd.DataFrame
    acceptance: dict[str, object]


def rolling_fold_ranges(first_test: date, last_day: date, window_days: int) -> list[tuple[date, date]]:
    """Return contiguous fixed-length validation folds without duplicated dates.

    The test windows begin immediately after the initial training period and
    advance by their own length.  This ensures every held-out station/date is
    scored at most once, unlike monthly starts paired with fixed 31-day spans.
    """
    if window_days < 1:
        raise ValueError("test_window_days must be at least one.")
    ranges: list[tuple[date, date]] = []
    start = first_test
    while start <= last_day:
        end = min(last_day, start + timedelta(days=window_days - 1))
        ranges.append((start, end))
        start = end + timedelta(days=1)
    return ranges


def _metric_row(data: pd.DataFrame, forecast_column: str, label: str) -> dict[str, object]:
    output: dict[str, object] = {"model": label}
    output.update(forecast_metrics(data, TARGET_COLUMN, forecast_column))
    if forecast_column != BASELINE_COLUMN:
        output["mae_skill_vs_nbm"] = relative_mae_skill(data, TARGET_COLUMN, forecast_column, BASELINE_COLUMN)
    else:
        output["mae_skill_vs_nbm"] = 0.0
    interval_columns = {
        "ridge_prediction_f": ("ridge_interval_lower_f", "ridge_interval_upper_f"),
        "xgb_prediction_f": ("interval_lower_f", "interval_upper_f"),
        "blend_prediction_f": ("blend_interval_lower_f", "blend_interval_upper_f"),
    }
    lower_upper = interval_columns.get(forecast_column)
    if lower_upper and set(lower_upper).issubset(data.columns):
        output.update(interval_metrics(data, lower=lower_upper[0], upper=lower_upper[1]))
    return output


def _bootstrap_metrics(
    data: pd.DataFrame,
    repeats: int = 400,
    random_state: int = 20260813,
) -> pd.DataFrame:
    """Block bootstrap resampling entire forecast dates, not pseudo-independent hours."""
    rng = np.random.default_rng(random_state)
    day_groups = [block for _, block in data.groupby("target_date", sort=False)]
    if len(day_groups) < 10:
        return pd.DataFrame()
    rows = []
    for _ in range(repeats):
        sampled = pd.concat([day_groups[index] for index in rng.integers(0, len(day_groups), len(day_groups))])
        for column, label in ((BASELINE_COLUMN, "NBM"), ("ridge_prediction_f", "Ridge residual"), ("xgb_prediction_f", "XGBoost residual"), ("blend_prediction_f", "Station blend residual")):
            metrics = _metric_row(sampled, column, label)
            rows.append({"model": label, "mae_f": metrics["mae_f"], "mae_skill_vs_nbm": metrics["mae_skill_vs_nbm"]})
    distribution = pd.DataFrame(rows)
    return (
        distribution.groupby("model")
        .agg(
            mae_ci_low=("mae_f", lambda x: float(x.quantile(0.025))),
            mae_ci_high=("mae_f", lambda x: float(x.quantile(0.975))),
            skill_ci_low=("mae_skill_vs_nbm", lambda x: float(x.quantile(0.025))),
            skill_ci_high=("mae_skill_vs_nbm", lambda x: float(x.quantile(0.975))),
        )
        .reset_index()
    )


def _acceptance(
    predictions: pd.DataFrame,
    by_station: pd.DataFrame,
    expected_stations: Iterable[str] | None = None,
    bootstrap: pd.DataFrame | None = None,
) -> dict[str, object]:
    # XGBoost is predeclared as the production candidate. Choosing whichever
    # challenger happened to win on these same scored folds would make the
    # reported test set part of model selection and bias the result.
    overall_xgb = _metric_row(predictions, "xgb_prediction_f", "XGBoost residual")
    overall_blend = _metric_row(predictions, "blend_prediction_f", "Station blend residual")
    overall_ridge = _metric_row(predictions, "ridge_prediction_f", "Ridge residual")
    candidate_label = "XGBoost residual"
    candidate_column = "xgb_prediction_f"
    candidate = overall_xgb
    total = by_station.loc[by_station["model"] == candidate_label].copy()
    expected = {str(station).upper() for station in (expected_stations if expected_stations is not None else STATIONS)}
    evaluated = {str(station).upper() for station in total["station"].unique()}
    missing_stations = sorted(expected - evaluated)
    station_wins = total.loc[
        total["station"].astype(str).str.upper().isin(expected) & (total["mae_skill_vs_nbm"] > 0), "station"
    ].nunique()
    margin = 0.01
    condition_nbm = bool(candidate["mae_skill_vs_nbm"] >= margin)
    condition_ridge = bool(candidate["mae_f"] < overall_ridge["mae_f"])
    # A 20-city release cannot pass because a small subset performed well.
    # Every configured station must be evaluated and beat raw NBM.
    condition_stations = bool(not missing_stations and station_wins >= len(expected))
    coverage_floor = CONFORMAL_NOMINAL_COVERAGE - 0.02
    severe_station_coverage_floor = CONFORMAL_NOMINAL_COVERAGE - 0.05
    overall_coverage = float(candidate.get("coverage", np.nan))
    condition_interval = bool(np.isfinite(overall_coverage) and overall_coverage >= coverage_floor)
    station_coverage = pd.to_numeric(total.get("coverage", pd.Series(dtype=float)), errors="coerce")
    condition_station_intervals = bool(
        len(station_coverage) == len(expected)
        and station_coverage.notna().all()
        and (station_coverage >= severe_station_coverage_floor).all()
    )
    bootstrap_skill_low = np.nan
    if bootstrap is not None and not bootstrap.empty:
        bootstrap_row = bootstrap.loc[bootstrap["model"].eq(candidate_label)]
        if not bootstrap_row.empty:
            bootstrap_skill_low = float(bootstrap_row.iloc[0]["skill_ci_low"])
    condition_bootstrap = bool(np.isfinite(bootstrap_skill_low) and bootstrap_skill_low > 0)
    statistical_candidate = bool(
        condition_nbm
        and condition_ridge
        and condition_stations
        and condition_interval
        and condition_station_intervals
        and condition_bootstrap
    )
    return {
        # Statistical skill is necessary but not sufficient. The project's
        # present archived source reconstructs every hourly value at a fixed
        # lead; it does not reproduce one frozen daily issuance. The PDF's
        # central release gate therefore requires prospective shadow logging.
        "decision": "SHADOW_ONLY",
        "statistical_candidate_passed": statistical_candidate,
        "minimum_global_mae_skill_vs_nbm": margin,
        "xgb_mae_skill_vs_nbm": overall_xgb["mae_skill_vs_nbm"],
        "xgb_mae_f": overall_xgb["mae_f"],
        "blend_mae_skill_vs_nbm": overall_blend["mae_skill_vs_nbm"],
        "blend_mae_f": overall_blend["mae_f"],
        "candidate_model": candidate_label,
        "candidate_prediction_column": candidate_column,
        "candidate_mae_f": candidate["mae_f"],
        "candidate_mae_skill_vs_nbm": candidate["mae_skill_vs_nbm"],
        "candidate_coverage": overall_coverage,
        "required_overall_coverage": coverage_floor,
        "required_min_station_coverage": severe_station_coverage_floor,
        "interval_gate_passed": condition_interval,
        "station_interval_gate_passed": condition_station_intervals,
        "bootstrap_skill_ci_low": bootstrap_skill_low,
        "bootstrap_skill_gate_passed": condition_bootstrap,
        "ridge_mae_f": overall_ridge["mae_f"],
        "station_wins_vs_nbm": int(station_wins),
        "required_station_wins": len(expected),
        "expected_stations": sorted(expected),
        "evaluated_stations": sorted(evaluated),
        "missing_expected_stations": missing_stations,
        "reason": (
            "The XGBoost candidate cleared the rolling-backtest comparison gates, but remains shadow-only "
            "until forecasts are prospectively logged from one frozen issue-time contract and re-verified."
            if statistical_candidate
            else "Retain in shadow mode: at least one rolling-backtest NBM/Ridge/station gate was not met."
        ),
        "release_blocker": "The retrospective predictor is an hour-wise 24-hour-lead composite, not a single frozen daily forecast issuance."
        " Log and score prospective fixed-cutoff forecasts before production promotion.",
    }


def run_rolling_backtest(
    table: pd.DataFrame,
    initial_train_days: int = 180,
    test_window_days: int = 31,
    calibration_days: int = 60,
    training_window_days: int | None = None,
    expected_stations: Iterable[str] | None = None,
) -> BacktestResult:
    """Evaluate models using expanding, calendar-ordered folds only."""
    data = table.copy()
    data["target_date"] = pd.to_datetime(data["target_date"]).dt.date
    data = data.dropna(subset=[TARGET_COLUMN, BASELINE_COLUMN]).sort_values("target_date")
    first_day = min(data["target_date"])
    last_day = max(data["target_date"])
    first_test = first_day + timedelta(days=initial_train_days)
    predictions: list[pd.DataFrame] = []
    for fold_start, fold_end in rolling_fold_ranges(first_test, last_day, test_window_days):
        train = data.loc[data["target_date"] < fold_start].copy()
        if training_window_days is not None:
            window_start = fold_start - timedelta(days=training_window_days)
            train = train.loc[train["target_date"] >= window_start].copy()
        test = data.loc[(data["target_date"] >= fold_start) & (data["target_date"] <= fold_end)].copy()
        if train.empty or test.empty:
            continue
        climatology = SeasonalClimatology.fit(train)
        test["climatology_f"] = climatology.predict(test)
        ridge = ResidualForecaster.fit(train, "ridge", calibration_days=calibration_days)
        xgb = ResidualForecaster.fit(train, "xgb", calibration_days=calibration_days)
        blend = BlendedResidualForecaster.fit(train, calibration_days=calibration_days)
        ridge_output = ridge.predict(test).rename(
            columns={
                "prediction_f": "ridge_prediction_f",
                "interval_lower_f": "ridge_interval_lower_f",
                "interval_upper_f": "ridge_interval_upper_f",
            }
        )
        xgb_output = xgb.predict(test).rename(columns={"prediction_f": "xgb_prediction_f"})
        blend_output = blend.predict(test).rename(
            columns={
                "prediction_f": "blend_prediction_f",
                "interval_lower_f": "blend_interval_lower_f",
                "interval_upper_f": "blend_interval_upper_f",
            }
        )
        test = pd.concat(
            [
                test.reset_index(drop=True),
                ridge_output[["ridge_prediction_f", "ridge_interval_lower_f", "ridge_interval_upper_f"]].reset_index(drop=True),
                xgb_output.reset_index(drop=True),
                blend_output[["blend_prediction_f", "blend_interval_lower_f", "blend_interval_upper_f"]].reset_index(drop=True),
            ],
            axis=1,
        )
        test["fold_start"] = fold_start
        test["fold_end"] = fold_end
        test["train_rows"] = len(train)
        predictions.append(test)
    if not predictions:
        raise ValueError("No rolling backtest folds could be formed from the training table.")
    result = pd.concat(predictions, ignore_index=True)
    model_columns = (
        ("climatology_f", "Seasonal climatology"),
        (BASELINE_COLUMN, "NBM"),
        ("ridge_prediction_f", "Ridge residual"),
        ("xgb_prediction_f", "XGBoost residual"),
        ("blend_prediction_f", "Station blend residual"),
    )
    overall = pd.DataFrame([_metric_row(result, column, label) for column, label in model_columns])
    by_station_rows: list[dict[str, object]] = []
    for station, group in result.groupby("station", sort=True):
        for column, label in model_columns:
            row = _metric_row(group, column, label)
            row["station"] = station
            by_station_rows.append(row)
    by_station = pd.DataFrame(by_station_rows)
    by_month_rows: list[dict[str, object]] = []
    result["month"] = pd.to_datetime(result["target_date"]).dt.month
    for month, group in result.groupby("month", sort=True):
        for column, label in model_columns:
            row = _metric_row(group, column, label)
            row["month"] = int(month)
            by_month_rows.append(row)
    by_month = pd.DataFrame(by_month_rows)
    bootstrap = _bootstrap_metrics(result)
    return BacktestResult(
        predictions=result,
        overall=overall,
        by_station=by_station,
        by_month=by_month,
        bootstrap=bootstrap,
        acceptance=_acceptance(result, by_station, expected_stations=expected_stations, bootstrap=bootstrap),
    )


def write_backtest(result: BacktestResult, output_dir: str | Path) -> Path:
    """Write reproducible prediction-level and summary artifacts."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(directory / "rolling_predictions.parquet", index=False)
    result.overall.to_csv(directory / "overall_metrics.csv", index=False)
    result.by_station.to_csv(directory / "station_metrics.csv", index=False)
    result.by_month.to_csv(directory / "month_metrics.csv", index=False)
    result.bootstrap.to_csv(directory / "block_bootstrap_ci.csv", index=False)
    (directory / "acceptance.json").write_text(
        pd.Series(result.acceptance).to_json(indent=2), encoding="utf-8"
    )
    return directory
