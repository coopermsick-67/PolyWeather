"""Export the validated rolling-backtest rows used by the browser workspace.

The app deliberately evaluates the same immutable held-out rows as the model
report. It does not synthesize outcomes or imply an on-demand forecast replay.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "artifacts" / "backtest_v4" / "rolling_predictions.parquet"
ACCEPTANCE = ROOT / "artifacts" / "backtest_v4" / "acceptance.json"
OUTPUT = ROOT / "dashboard" / "public" / "backtest-data.json"

FIELDS = {
    "station": "station",
    "target_date": "date",
    "tmax_f": "observedF",
    "nbm_baseline_f": "nbmF",
    "ncep_hrrr_conus__tmax_f": "hrrrF",
    "ncep_gfs_seamless__tmax_f": "gfsF",
    "ridge_prediction_f": "ridgeF",
    "xgb_prediction_f": "xgbF",
    "blend_prediction_f": "blendF",
    # The existing chart keys are retained for compatibility, but the shaded
    # band is the coverage-calibrated interval.  True conditional tail
    # estimates are exported separately and are not mislabeled as the band.
    "interval_lower_f": "p10F",
    "interval_upper_f": "p90F",
    "p10_f": "quantileP10F",
    "p90_f": "quantileP90F",
}


def main() -> None:
    table = pq.read_table(INPUT, columns=list(FIELDS))
    records = []
    for row in table.to_pylist():
        record = {}
        for source, target in FIELDS.items():
            value = row[source]
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            elif isinstance(value, float):
                value = round(value, 3)
            record[target] = value
        if all(isinstance(record[key], (int, float)) for key in ("nbmF", "hrrrF", "gfsF")):
            record["consensusF"] = round(.5 * record["nbmF"] + .25 * record["hrrrF"] + .25 * record["gfsF"], 3)
        records.append(record)
    records.sort(key=lambda record: (record["station"], record["date"]))
    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    archive_fingerprint = hashlib.sha256(json.dumps(records, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()[:12]
    payload = {
        "version": "backtest_v4_20station",
        "fingerprint": archive_fingerprint,
        "status": "SHADOW_ONLY",
        "target": "Official NOAA/NCEI local-day TMAX",
        "contract": "One archived 24-hour lead per valid hour, aggregated to each station's local-day high.",
        "limitation": acceptance["release_blocker"],
        "liveParity": "This frozen 20-station cohort does not prove live-model accuracy. It is a retrospective, fixed-history evaluation and is not an on-demand replay.",
        "sources": ["Archived Open-Meteo NBM", "Archived Open-Meteo HRRR", "Archived Open-Meteo GFS", "Official NOAA/NCEI TMAX outcome"],
        "decision": acceptance["decision"],
        "dateRange": {"start": min(row["date"] for row in records), "end": max(row["date"] for row in records)},
        "stations": sorted({row["station"] for row in records}),
        "rows": records,
    }
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
