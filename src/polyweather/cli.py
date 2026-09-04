"""Command-line interface for reproducible PolyWeather workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from .backtest import run_rolling_backtest, write_backtest
from .data import add_derived_forecast_features, build_training_table, fetch_live_forecast_features, write_training_table
from .model import CONFORMAL_NOMINAL_COVERAGE, AdaptiveResidualForecaster, BlendedResidualForecaster, ResidualForecaster
from .quality import write_quality_report
from .reporting import build_backtest_charts, make_model_card
from .settlement import illustrative_report_by_station
from .shadow import append_jsonl, create_shadow_records, verify_shadow_log
from .stations import STATIONS, require_station


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Dates must use YYYY-MM-DD.") from exc


def _read_table(path: str | Path) -> pd.DataFrame:
    table = pd.read_parquet(path)
    table["target_date"] = pd.to_datetime(table["target_date"]).dt.date
    return add_derived_forecast_features(table)


def _requested_stations(requested: list[str], model_path: str | Path) -> list[str]:
    """Resolve ``all`` from the model manifest, never from unknown categories."""
    if requested != ["all"]:
        return [item.upper() for item in requested]
    manifest_path = Path(model_path).with_name("model_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_stations = [str(item).upper() for item in manifest["stations"]]
        return [icao for icao in model_stations if icao in STATIONS]
    except (OSError, ValueError, KeyError, TypeError):
        return list(STATIONS)


def command_build_data(args: argparse.Namespace) -> None:
    table = build_training_table(
        args.start,
        args.end,
        lead_days=args.lead_days,
        chunk_days=args.chunk_days,
        max_workers=args.max_workers,
        request_interval_seconds=args.request_interval_seconds,
        cache_dir=args.cache_dir,
    )
    output = write_training_table(table, args.output)
    print(json.dumps({"output": str(output), "rows": len(table), "stations": sorted(table.station.unique())}, indent=2))


def command_backtest(args: argparse.Namespace) -> None:
    table = _read_table(args.data)
    result = run_rolling_backtest(
        table,
        initial_train_days=args.initial_train_days,
        test_window_days=args.test_window_days,
        calibration_days=args.calibration_days,
        training_window_days=args.training_window_days,
    )
    directory = write_backtest(result, args.output_dir)
    print(result.overall.round(4).to_string(index=False))
    print(json.dumps(result.acceptance, indent=2))
    print(f"Wrote backtest artifacts to {directory}")


def command_train(args: argparse.Namespace) -> None:
    table = _read_table(args.data)
    if args.cutoff:
        table = table.loc[table["target_date"] <= args.cutoff].copy()
    if args.training_window_days:
        end_date = max(table["target_date"])
        table = table.loc[table["target_date"] >= end_date - timedelta(days=args.training_window_days)].copy()
    if args.kind == "adaptive":
        model = AdaptiveResidualForecaster.fit(table, calibration_days=args.calibration_days)
    elif args.kind == "blend":
        model = BlendedResidualForecaster.fit(table, calibration_days=args.calibration_days)
    else:
        model = ResidualForecaster.fit(table, args.kind, calibration_days=args.calibration_days)
    data_end = max(table["target_date"])
    calibration_start = (
        data_end - timedelta(days=args.calibration_days - 1) if model.calibration_rows else None
    )
    model_fit_end = calibration_start - timedelta(days=1) if calibration_start else data_end
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.kind}_residual_tmax.joblib"
    joblib.dump(model, model_path)
    manifest = {
        "model_path": str(model_path),
        "model_family": f"{args.kind.upper()} residual MOS",
        "target": "NCEI daily-summaries TMAX (°F)",
        "base_forecast": "NCEP NBM hourly profile maximum (°F)",
        "feature_schema_version": "v003_20station_regime_features",
        "training_rows": model.train_rows,
        "training_start": str(table.target_date.min()),
        "model_fit_end": str(model_fit_end),
        "training_data_end": str(data_end),
        "stations": sorted(table.station.unique().tolist()),
        "conformal_nominal_coverage": CONFORMAL_NOMINAL_COVERAGE,
        "conformal_halfwidth_f": model.conformal_halfwidth_f,
        # The quantile level actually applied after self-correction for
        # undercoverage (see CONFORMAL_SAFETY_MARGIN) -- distinct from the
        # nominal coverage above, which never changes. Only ResidualForecaster
        # carries this directly; adaptive/blend report their XGB member's
        # level as a representative value since both members self-correct
        # independently.
        "conformal_quantile_level_applied": getattr(
            model, "conformal_quantile_level", getattr(getattr(model, "xgb", None), "conformal_quantile_level", None)
        ),
        "station_model_selection": getattr(model, "station_models", None),
        "station_ridge_weights": getattr(model, "station_ridge_weights", None),
        "selection_rows": getattr(model, "selection_rows", None),
        "calibration_rows": model.calibration_rows,
        "calibration_start": str(calibration_start) if calibration_start else None,
        "calibration_end": str(data_end) if calibration_start else None,
        "issue_time_contract": "fixed 24h archived lead composite in backtest; current latest forecast at inference",
        "release_status": "SHADOW_ONLY until prospective fixed-cutoff logging and verification pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def command_predict(args: argparse.Namespace) -> None:
    # args.model is a local CLI flag supplied by the operator running this
    # command, not remote/request-controlled input; joblib.load still runs
    # pickle deserialization, so only ever point --model at trusted artifacts.
    model = joblib.load(args.model)
    target_date = args.target_date or (date.today() + timedelta(days=1))
    requested = _requested_stations(args.stations, args.model)
    rows = []
    for icao in requested:
        station = require_station(icao)
        features = fetch_live_forecast_features(station, target_date)
        forecast = model.predict(pd.DataFrame([features])).iloc[0].to_dict()
        rows.append(
            {
                "station": station.icao,
                "name": station.name,
                "target_date": str(target_date),
                "forecast_tmax_f": round(float(forecast["prediction_f"]), 1),
                "p10_f": round(float(forecast["p10_f"]), 1),
                "p50_f": round(float(forecast["p50_f"]), 1),
                "p90_f": round(float(forecast["p90_f"]), 1),
                "nbm_baseline_f": round(float(features["nbm_baseline_f"]), 1),
                "model": f"{model.kind.upper()} residual MOS",
                "interval": "nominal 75% split-conformal interval",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    print(json.dumps(rows, indent=2))


def command_report(args: argparse.Namespace) -> None:
    charts = build_backtest_charts(args.backtest_dir, args.output_dir)
    card = make_model_card(args.backtest_dir, Path(args.output_dir) / "MODEL_CARD.md")
    print(json.dumps({"model_card": str(card), "charts": [str(chart) for chart in charts]}, indent=2))


def command_quality(args: argparse.Namespace) -> None:
    table = _read_table(args.data)
    output = write_quality_report(table, args.output_dir)
    print(output.read_text(encoding="utf-8"))


def command_log_current(args: argparse.Namespace) -> None:
    # args.model is a local CLI flag supplied by the operator running this
    # command, not remote/request-controlled input; joblib.load still runs
    # pickle deserialization, so only ever point --model at trusted artifacts.
    model = joblib.load(args.model)
    target_date = args.target_date or (date.today() + timedelta(days=1))
    requested = _requested_stations(args.stations, args.model)
    model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    records = create_shadow_records(
        model,
        [require_station(icao) for icao in requested],
        target_date,
        model_artifact_sha256=model_hash,
    )
    for record in records:
        record.update(
            {
                "model_artifact_sha256": model_hash,
                "feature_schema_version": "v003_20station_regime_features",
                "model_train_rows": model.train_rows,
                "model_calibration_rows": model.calibration_rows,
            }
        )
    output = append_jsonl(records, args.log)
    print(json.dumps({"log": str(output), "records": records}, indent=2))


def command_illustrative_bucket_hit_rate(args: argparse.Namespace) -> None:
    """Research-only: illustrative/UNVERIFIED bucket-hit-rate + calibration report.

    This scores against `EXAMPLE_BUCKET_WIDTHS` (or a caller-supplied bucket
    width) because no real, verified `MarketContract` is populated in this
    repo yet -- every field in the report is prefixed `illustrative_` and it
    must never be read as a real, contract-backed hit rate. It does not
    touch, and is not read by, the live dashboard's NO_BET/dataQualityStatus
    gate in `dashboard_payload.py`.
    """
    predictions = pd.read_parquet(args.predictions)
    report = illustrative_report_by_station(predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "illustrative_bucket_hit_rate.json"
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote illustrative (UNVERIFIED) bucket-hit-rate report to {output_path}")


def command_verify_shadow(args: argparse.Namespace) -> None:
    verified, metrics = verify_shadow_log(args.log)
    if not verified.empty:
        verified.to_parquet(args.output, index=False)
    print(json.dumps({"verified_rows": len(verified), "metrics": metrics, "output": args.output}, indent=2, default=float))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolyWeather daily Tmax forecast system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-data", help="Download labels and archived forecast features")
    build.add_argument("--start", type=_date, required=True)
    build.add_argument("--end", type=_date, required=True)
    build.add_argument("--lead-days", type=int, default=1, choices=(1, 2, 3))
    build.add_argument("--chunk-days", type=int, default=31, help="Historical days per source request; smaller values are safer for public APIs.")
    build.add_argument("--max-workers", type=int, default=2, help="Bounded concurrent source requests.")
    build.add_argument("--request-interval-seconds", type=float, default=0.8, help="Process-wide delay between archived-source requests.")
    build.add_argument("--cache-dir", default="data/raw/open_meteo_previous_runs", help="Persistent raw-response cache; reruns resume without refetching completed chunks.")
    build.add_argument("--output", default="data/features/tmax_training.parquet")
    build.set_defaults(func=command_build_data)

    backtest = subparsers.add_parser("backtest", help="Run expanding rolling-origin validation")
    backtest.add_argument("--data", default="data/features/tmax_training.parquet")
    backtest.add_argument("--output-dir", default="artifacts/backtest")
    backtest.add_argument("--initial-train-days", type=int, default=180)
    backtest.add_argument("--test-window-days", type=int, default=31)
    backtest.add_argument("--calibration-days", type=int, default=60)
    backtest.add_argument("--training-window-days", type=int, help="Optional rolling training window; default is expanding.")
    backtest.set_defaults(func=command_backtest)

    train = subparsers.add_parser("train", help="Fit and serialize the final residual model")
    train.add_argument("--data", default="data/features/tmax_training.parquet")
    train.add_argument("--output-dir", default="artifacts/production")
    train.add_argument("--kind", choices=("ridge", "xgb", "adaptive", "blend"), default="blend")
    train.add_argument("--cutoff", type=_date)
    train.add_argument("--calibration-days", type=int, default=60)
    train.add_argument("--training-window-days", type=int, help="Optional rolling training window before the cutoff.")
    train.set_defaults(func=command_train)

    predict = subparsers.add_parser("predict", help="Generate current local-date Tmax forecasts")
    predict.add_argument("--model", default="artifacts/production/xgb_residual_tmax.joblib")
    predict.add_argument("--date", dest="target_date", type=_date)
    predict.add_argument("--stations", nargs="+", default=["all"], help="Station codes or 'all'")
    predict.set_defaults(func=command_predict)

    report = subparsers.add_parser("report", help="Build model-card and QA chart artifacts")
    report.add_argument("--backtest-dir", default="artifacts/backtest")
    report.add_argument("--output-dir", default="reports")
    report.set_defaults(func=command_report)

    quality = subparsers.add_parser("quality", help="Run high-signal training-table checks")
    quality.add_argument("--data", default="data/features/tmax_training.parquet")
    quality.add_argument("--output-dir", default="artifacts/quality")
    quality.set_defaults(func=command_quality)

    log_current = subparsers.add_parser("log-current", help="Append immutable prospective forecast snapshots")
    log_current.add_argument("--model", default="artifacts/production/xgb_residual_tmax.joblib")
    log_current.add_argument("--date", dest="target_date", type=_date)
    log_current.add_argument("--stations", nargs="+", default=["all"])
    log_current.add_argument("--log", default="data/normalized/shadow_forecasts.jsonl")
    log_current.set_defaults(func=command_log_current)

    verify = subparsers.add_parser("verify-shadow", help="Verify mature shadow snapshots against NCEI daily Tmax")
    verify.add_argument("--log", default="data/normalized/shadow_forecasts.jsonl")
    verify.add_argument("--output", default="artifacts/shadow/verified_forecasts.parquet")
    verify.set_defaults(func=command_verify_shadow)

    illustrative_hit_rate = subparsers.add_parser(
        "illustrative-bucket-hit-rate",
        help="Research-only: illustrative/UNVERIFIED bucket-hit-rate + calibration report (no real settlement contract).",
    )
    illustrative_hit_rate.add_argument("--predictions", default="artifacts/backtest_20_enhanced/rolling_predictions.parquet")
    illustrative_hit_rate.add_argument("--output-dir", default="artifacts/settlement")
    illustrative_hit_rate.set_defaults(func=command_illustrative_bucket_hit_rate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
