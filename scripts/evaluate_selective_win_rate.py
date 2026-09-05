"""Measure whether an 85% win rate is reachable, and on which contract type.

Selectivity is the only honest lever this system has: the weather model's
skill is fixed, so the question is not "can the forecast be made 85%
accurate" but "is there a subset of city-days, identifiable *before*
settlement, on which some contract settles our way 85%+ of the time".

Protocol, so the answer is not manufactured by tuning:
  * Every threshold is chosen on the VALIDATION window only.
  * The frozen thresholds are then applied once to the TEST window.
  * A validation result that fails to reproduce on test is reported as a
    failure, not re-tuned until it passes.

Both windows use out-of-sample predictions produced by
``evaluate_accuracy_candidates.py``: the validation predictions come from a
model fit strictly before the validation window, the test predictions from a
model fit strictly before the test window.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polyweather.bet_evidence import bucket_for
from polyweather.betfilter.results import settled_in_bucket

PREDICTION = "current_xgb"
TARGET = 0.85
# Below this share of city-days a "win rate" rests on a handful of rows, and
# its confidence interval is too wide to tell 85% from 60%.
MIN_COVERAGE = 0.05
SEED = 20260905

# Signals a forecaster genuinely holds before settlement. Nothing derived
# from the observed high may appear here.
CONFIDENCE_FEATURES = [
    "model_agreement__tmax_f_spread",
    "model_agreement__tmax_f_std",
    "model_agreement__nbm_minus_hrrr__tmax_f",
    "model_agreement__nbm_position_tmax_f",
    "forecast_lead_days",
]


def _outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    error = frame[PREDICTION].to_numpy(float) - frame["tmax_f"].to_numpy(float)
    frame["abs_error_f"] = np.abs(error)
    frame["within_2f"] = frame["abs_error_f"] <= 2.0
    frame["bucket_hit"] = [
        settled_in_bucket(actual, *bucket_for(predicted))
        for actual, predicted in zip(frame["tmax_f"], frame[PREDICTION], strict=True)
    ]
    return frame


def block_ci(frame: pd.DataFrame, column: str, seed: int = SEED) -> list[float] | None:
    """Weekly-block bootstrap CI. Same-day forecasts across cities share one
    weather pattern, so an i.i.d. interval would be far too narrow."""
    if frame.empty:
        return None
    daily = frame.groupby("target_date")[column].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    n = len(daily)
    draws = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=int(np.ceil(n / 7)))
        index = ((starts[:, None] + np.arange(7)) % n).ravel()[:n]
        sample = daily.iloc[index]
        draws.append(float(sample["sum"].sum() / sample["count"].sum()))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def _report(frame: pd.DataFrame, column: str, total: int) -> dict:
    if frame.empty:
        return {"n": 0, "coverage": 0.0, "win_rate": None,
                "win_rate_95ci": None, "target_met": False}
    ci = block_ci(frame, column)
    coverage = len(frame) / total
    return {
        "n": int(len(frame)),
        "coverage": float(coverage),
        "win_rate": float(frame[column].mean()),
        "win_rate_95ci": ci,
        # The lower bound must clear the target: a point estimate of 0.86 on
        # 40 rows is not evidence of an 85% edge.
        "target_met": bool(ci and ci[0] >= TARGET and coverage >= MIN_COVERAGE),
    }


def selective_confidence(validation: pd.DataFrame, test: pd.DataFrame, outcome: str) -> dict:
    """Fit a confidence model on validation, freeze a cutoff, apply to test."""
    model = Pipeline([("scale", StandardScaler()),
                      ("model", LogisticRegression(C=0.5, max_iter=2000, random_state=SEED))])
    model.fit(validation[CONFIDENCE_FEATURES].astype(float), validation[outcome].astype(int))
    validation = validation.assign(confidence=model.predict_proba(
        validation[CONFIDENCE_FEATURES].astype(float))[:, 1])
    test = test.assign(confidence=model.predict_proba(
        test[CONFIDENCE_FEATURES].astype(float))[:, 1])

    # Lowest cutoff (so, widest coverage) that reaches the target on validation.
    frozen, curve = None, []
    for cutoff in np.round(np.arange(0.30, 0.99, 0.01), 2):
        subset = validation[validation.confidence >= cutoff]
        coverage = len(subset) / len(validation)
        rate = float(subset[outcome].mean()) if len(subset) else None
        curve.append({"cutoff": float(cutoff), "n": int(len(subset)),
                      "coverage": float(coverage), "win_rate": rate})
        if frozen is None and rate is not None and rate >= TARGET and coverage >= MIN_COVERAGE:
            frozen = float(cutoff)

    eligible = [row for row in curve if row["coverage"] >= MIN_COVERAGE and row["win_rate"] is not None]
    result = {
        "outcome": outcome,
        "frozen_cutoff": frozen,
        "validation_best_achievable": max(eligible, key=lambda row: row["win_rate"], default=None),
        "validation_curve": curve,
    }
    if frozen is None:
        result["test"] = None
        result["conclusion"] = (
            f"No confidence cutoff reached {TARGET:.0%} on validation at >={MIN_COVERAGE:.0%} "
            "coverage, so nothing was carried forward to test.")
        return result
    result["test"] = _report(test[test.confidence >= frozen], outcome, len(test))
    return result


def threshold_contracts(validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Win rate for THRESHOLD_GTE/LTE contracts set a margin off the forecast.

    A bracket asks the forecast to be nearly exact. A threshold contract only
    asks it to land on the right side of a line, so pushing the line away
    from the forecast buys win rate directly -- at whatever price the market
    charges for it, which this analysis does not model.
    """
    output = {
        "note": "Win rate only. No market pricing here, so this is not an edge or a profit claim.",
        "margins": [], "frozen_margin_f": None, "test": None,
    }
    for margin in np.arange(0.0, 12.5, 0.5):
        row = {"margin_f": float(margin)}
        for name, frame in (("validation", validation), ("test", test)):
            actual = frame["tmax_f"].to_numpy(float)
            predicted = frame[PREDICTION].to_numpy(float)
            row[f"{name}_gte_win_rate"] = float(np.mean(actual >= predicted - margin))
            row[f"{name}_lte_win_rate"] = float(np.mean(actual <= predicted + margin))
        output["margins"].append(row)

    # Freeze the smallest margin clearing the target on BOTH sides in validation.
    for row in output["margins"]:
        if row["validation_gte_win_rate"] >= TARGET and row["validation_lte_win_rate"] >= TARGET:
            output["frozen_margin_f"] = row["margin_f"]
            break
    if output["frozen_margin_f"] is None:
        return output

    margin = output["frozen_margin_f"]
    frame = test.copy()
    frame["gte_win"] = frame["tmax_f"] >= frame[PREDICTION] - margin
    frame["lte_win"] = frame["tmax_f"] <= frame[PREDICTION] + margin
    output["test"] = {"margin_f": margin,
                      "gte": _report(frame, "gte_win", len(frame)),
                      "lte": _report(frame, "lte_win", len(frame))}
    return output


def run(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    validation = _outcomes(pd.read_parquet(source / "validation_predictions.parquet"))
    test = _outcomes(pd.read_parquet(source / "test_predictions.parquet"))

    result = {
        "protocol": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "validation_window": [str(validation.target_date.min().date()),
                                  str(validation.target_date.max().date())],
            "test_window": [str(test.target_date.min().date()),
                            str(test.target_date.max().date())],
            "target": TARGET,
            "minimum_coverage": MIN_COVERAGE,
            "rule": ("Thresholds frozen on validation, measured once on test; "
                     "the 95% CI lower bound must clear the target"),
            "evidence_status": ("RETROSPECTIVE_RESEARCH; the archive was inspected during "
                                "development and issuance times do not match live operation; "
                                "not betting evidence"),
        },
        "unconditional": {
            "validation": {"within_2f": float(validation.within_2f.mean()),
                           "bucket_hit_rate": float(validation.bucket_hit.mean())},
            "test": {"within_2f": float(test.within_2f.mean()),
                     "bucket_hit_rate": float(test.bucket_hit.mean()),
                     "within_2f_95ci": block_ci(test, "within_2f"),
                     "bucket_hit_95ci": block_ci(test, "bucket_hit")},
        },
        "selective_within_2f": selective_confidence(validation, test, "within_2f"),
        "selective_bucket": selective_confidence(validation, test, "bucket_hit"),
        "threshold_contracts": threshold_contracts(validation, test),
    }
    (output / "selective_win_rate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    summary = {
        "unconditional": result["unconditional"],
        "selective_within_2f": {k: v for k, v in result["selective_within_2f"].items()
                                if k != "validation_curve"},
        "selective_bucket": {k: v for k, v in result["selective_bucket"].items()
                             if k != "validation_curve"},
    }
    print(json.dumps(summary, indent=2), flush=True)
    print("\n=== threshold contracts ===", flush=True)
    print(json.dumps({k: v for k, v in result["threshold_contracts"].items()
                      if k != "margins"}, indent=2), flush=True)
    for row in result["threshold_contracts"]["margins"]:
        if row["margin_f"] % 1 == 0:
            print("margin %4.1fF  val gte %.3f lte %.3f | test gte %.3f lte %.3f" % (
                row["margin_f"], row["validation_gte_win_rate"],
                row["validation_lte_win_rate"], row["test_gte_win_rate"],
                row["test_lte_win_rate"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("reports/accuracy_improvement_2026-09-05"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/accuracy_improvement_2026-09-05"))
    args = parser.parse_args()
    run(args.source, args.output)
