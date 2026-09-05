"""Frozen chronological research comparison; never replaces production artifacts.

The archive has already been used in development, so even the final chronological
test below is retrospective, not a pristine prospective validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from polyweather.bet_evidence import bucket_for
from polyweather.betfilter.results import settled_in_bucket
from polyweather.metrics import forecast_metrics
from polyweather.model import (
    ResidualForecaster, _prepare_feature_frame, _preprocessor,
    select_feature_columns, usable_training_rows, _validate_training_contract,
)


def candidate_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "current_xgb":
        return frame.copy()
    # Remove the aliased GFS feed and all agreement statistics that used it.
    # This is an ablation, not a claim to have repaired upstream source identity.
    return frame.drop(columns=[c for c in frame if c.startswith("ncep_gfs_seamless__")
                               or c.startswith("model_agreement__")])


def predict_candidate(train: pd.DataFrame, future: pd.DataFrame, name: str) -> np.ndarray:
    train, future = candidate_frame(train, name), candidate_frame(future, name)
    if name in {"current_xgb", "deduplicated_xgb"}:
        # Input-quality filtering was done on all three original feeds first.
        # Availability fields are audit metadata for this two-feed ablation.
        if name != "current_xgb":
            train = train.drop(columns=[c for c in train if c.endswith("__availability")])
        return ResidualForecaster.fit(train, "xgb").predict(future).prediction_f.to_numpy(float)
    cutoff = train.target_date.max() - pd.Timedelta(days=59)
    fit_rows, calibration = train[train.target_date < cutoff], train[train.target_date >= cutoff]
    categorical, numeric = select_feature_columns(fit_rows)
    estimator = Pipeline([
        ("features", _preprocessor(categorical, numeric, False)),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=300, learning_rate=.05,
            max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=10,
            early_stopping=False, random_state=20260905)),
    ])
    estimator.fit(_prepare_feature_frame(fit_rows, categorical, numeric),
                  fit_rows.tmax_f - fit_rows.nbm_baseline_f)
    errors = calibration.tmax_f.to_numpy() - calibration.nbm_baseline_f.to_numpy() - estimator.predict(
        _prepare_feature_frame(calibration, categorical, numeric))
    offsets = calibration.assign(error=errors).groupby("station").error.median()
    return (future.nbm_baseline_f.to_numpy() + estimator.predict(_prepare_feature_frame(future, categorical, numeric))
            + future.station.map(offsets).fillna(float(np.median(errors))).to_numpy())


def measure(rows: pd.DataFrame, name: str) -> dict:
    metric = forecast_metrics(rows, "tmax_f", name)
    if not len(rows):
        return {"n": 0, "within_2f": None, "default_bucket_hit_rate": None}
    metric["default_bucket_hit_rate"] = float(np.mean([
        settled_in_bucket(actual, *bucket_for(predicted))
        for actual, predicted in zip(rows.tmax_f, rows[name], strict=True)]))
    city_rates = rows.assign(hit=(rows[name] - rows.tmax_f).abs() <= 2).groupby("station").hit.mean()
    metric["city_average_within_2f"] = float(city_rates.mean())
    return metric


def block_interval(rows: pd.DataFrame, name: str, seed: int = 20260905) -> list[float] | None:
    if rows.empty:
        return None
    daily = rows.assign(hit=(rows[name] - rows.tmax_f).abs() <= 2).groupby("target_date").hit.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    n = len(daily)
    draws = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=int(np.ceil(n / 7)))
        indices = ((starts[:, None] + np.arange(7)) % n).ravel()[:n]
        sample = daily.iloc[indices]
        draws.append(float(sample["sum"].sum() / sample["count"].sum()))
    return np.quantile(draws, [.025, .975]).tolist()


def run(data: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(data)
    _validate_training_contract(raw)
    raw["target_date"] = pd.to_datetime(raw.target_date)
    frame = usable_training_rows(raw).sort_values(["target_date", "station"])
    test_start = frame.target_date.max() - pd.Timedelta(days=59)
    validation_start = test_start - pd.Timedelta(days=60)
    names = ["current_xgb", "deduplicated_xgb", "deduplicated_hist_absolute"]
    protocol = {
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "raw_rows": len(raw), "usable_rows": len(frame), "excluded_rows": len(raw) - len(frame),
        "validation_start": str(validation_start.date()), "test_start": str(test_start.date()),
        "test_end": str(frame.target_date.max().date()), "candidates": names,
        "selection": "Highest all-row within-2F on validation, then lowest MAE; fixed candidates, no test tuning",
        "station_selection": "Validation within-2F >=85%, at least 30 rows; fixed before test; minimum test coverage 10%",
        "target": .85, "minimum_selection_coverage": .10,
        "evidence_status": "RETROSPECTIVE_RESEARCH; previously inspected archive; mismatched live issuance; not betting evidence",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    validation = frame[(frame.target_date >= validation_start) & (frame.target_date < test_start)].copy()
    train = frame[frame.target_date < validation_start]
    validation_metrics = {}
    for name in names:
        print(f"Validation: fitting {name} on {len(train)} rows", flush=True)
        validation[name] = predict_candidate(train, validation, name)
        validation_metrics[name] = measure(validation, name)
        print(json.dumps(validation_metrics[name]), flush=True)
    chosen = max(names, key=lambda name: (validation_metrics[name]["within_2f"], -validation_metrics[name]["mae_f"]))
    station_stats = validation.assign(hit=(validation[chosen] - validation.tmax_f).abs() <= 2).groupby("station").hit.agg(["mean", "count"])
    stations = station_stats.index[(station_stats["mean"] >= .85) & (station_stats["count"] >= 30)].tolist()
    (output / "frozen_selection.json").write_text(json.dumps({"model": chosen, "stations": stations,
        "validation_metrics": validation_metrics}, indent=2), encoding="utf-8")
    test = frame[frame.target_date >= test_start].copy()
    test_metrics = {}
    for name in dict.fromkeys(["current_xgb", chosen]):
        print(f"Test: fitting frozen {name}", flush=True)
        test[name] = predict_candidate(frame[frame.target_date < test_start], test, name)
        test_metrics[name] = measure(test, name)
        test_metrics[name]["within_2f_block_95ci"] = block_interval(test, name)
    selected = test[test.station.isin(stations)]
    selected_metrics = measure(selected, chosen)
    selected_metrics["selection_coverage"] = len(selected) / len(test)
    selected_metrics["within_2f_block_95ci"] = block_interval(selected, chosen)
    ci = selected_metrics["within_2f_block_95ci"]
    selected_metrics["retrospective_85pct_lower_bound_met"] = bool(ci and ci[0] >= .85 and len(selected) / len(test) >= .10)
    result = {"protocol": protocol, "selected_candidate": chosen, "selected_stations": stations,
              "validation": validation_metrics, "test_all_stations": test_metrics,
              "test_selected_stations": selected_metrics, "production_promoted": False,
              "verified_operational_win_rate": None}
    validation.to_parquet(output / "validation_predictions.parquet", index=False)
    test.to_parquet(output / "test_predictions.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/features/tmax_24h_composite_training_v4.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/accuracy_improvement_2026-09-05"))
    args = parser.parse_args()
    run(args.data, args.output)
