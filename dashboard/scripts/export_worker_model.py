"""Export the production residual model in a Worker-readable JSON form."""

from __future__ import annotations

import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "production_v2" / "xgb_residual_tmax.joblib"
OUTPUT_PATH = ROOT / "dashboard" / "public" / "xgb-worker-model.json"


def main() -> None:
    model = joblib.load(MODEL_PATH)
    preprocess = model.pipeline.named_steps["preprocess"]
    imputer = preprocess.named_transformers_["numeric"].named_steps["impute"]
    onehot = preprocess.named_transformers_["station"].named_steps["onehot"]
    booster = model.pipeline.named_steps["model"].get_booster()
    payload = {
        "numericColumns": model.numeric_columns,
        "numericMedians": [float(value) for value in imputer.statistics_],
        "missingIndicatorIndices": [int(value) for value in imputer.indicator_.features_],
        "stationCategories": [str(value) for value in onehot.categories_[0]],
        "calibrationOffsetByStation": model.calibration_offset_by_station_f,
        "conformalHalfwidthByStation": model.conformal_halfwidth_by_station_f,
        "calibrationOffset": float(model.calibration_offset_f),
        "conformalHalfwidth": float(model.conformal_halfwidth_f),
        "booster": json.loads(booster.save_raw(raw_format="json")),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
