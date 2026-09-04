"""Refresh the dashboard's historical evidence from the 20-station backtest.

The worker imports this snapshot at build time.  Keeping the evidence derived
from the same immutable backtest artifacts prevents a stale UI label from
making newly trained stations appear unvalidated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "dashboard" / "public" / "dashboard-snapshot.json"
STATION_METRICS = ROOT / "artifacts" / "backtest_v4" / "station_metrics.csv"
OVERALL_METRICS = ROOT / "artifacts" / "backtest_v4" / "overall_metrics.csv"
MODEL = "XGBoost residual"


def metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    # The snapshot is now metadata/evidence only.  Shipping dated forecast
    # values here lets a network failure masquerade as a live prediction.
    snapshot["forecasts"] = []
    snapshot["generatedAt"] = None
    snapshot["targetDate"] = None
    snapshot["evaluationContract"] = "10,279 held-out station-day forecasts; archived 24-hour lead composite"
    snapshot["modelStatus"] = "SHADOW_ONLY v4; live values are never read from this file"
    stations = [row for row in metric_rows(STATION_METRICS) if row["model"] == MODEL]
    stations.sort(key=lambda row: number(row, "mae_f"))
    snapshot["accuracy"] = [
        {
            "rank": rank,
            "station": row["station"],
            "maeF": round(number(row, "mae_f"), 2),
            "p90ErrorF": round(number(row, "p90_ae_f"), 2),
            "within2Pct": round(number(row, "within_2f") * 100),
            "biasF": round(number(row, "bias_f"), 2),
        }
        for rank, row in enumerate(stations, start=1)
    ]

    overall = metric_rows(OVERALL_METRICS)
    candidate = next(row for row in overall if row["model"] == MODEL)
    baseline = next(row for row in overall if row["model"] == "NBM")
    snapshot["modelEvidence"] = {
        "candidateMaeF": round(number(candidate, "mae_f"), 2),
        "baselineMaeF": round(number(baseline, "mae_f"), 2),
        "skillPct": round(number(candidate, "mae_skill_vs_nbm") * 100),
        "within2Pct": round(number(candidate, "within_2f") * 100),
        "testForecasts": int(number(candidate, "n")),
        "fourDegreeCoveragePct": round(number(candidate, "within_2f") * 100),
        "calibratedCoveragePct": round(number(candidate, "coverage") * 100),
        "calibratedMeanWidthF": round(number(candidate, "mean_width_f"), 1),
    }
    SNAPSHOT.write_text(json.dumps(snapshot, separators=(",", ":"), allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
