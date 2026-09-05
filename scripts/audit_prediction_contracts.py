"""Offline, bounded reproductions of prediction/decision contract defects."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import numpy as np
import pandas as pd
import joblib
from polyweather import bet_evidence as be
from polyweather.betfilter.distribution import from_residual_history, candidate_buckets
from polyweather.betfilter.stability import ForecastSnapshot, analyze
from polyweather.model import feature_completeness
from polyweather.data import has_complete_core_guidance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/predictive_audit_2026-09-04"
OUT.mkdir(parents=True, exist_ok=True)
cal = be.probability_calibrator()
rows = []
for station, residuals in be.station_residuals().items():
    values = np.sort(residuals)
    # Any translation of a two-degree window is bounded by this scan.
    right = np.searchsorted(values, values + 2, side="left")
    raw = float(np.max(right - np.arange(len(values))) / len(values))
    rows.append({"station":station, "max_any_2F_raw_probability":raw,
                 "max_calibrated_probability":cal.apply(raw),
                 "residual_sigma":float(np.std(residuals,ddof=1))})
ceilings = pd.DataFrame(rows)
ceilings.to_csv(OUT / "probability_ceiling.csv",index=False)

distribution = from_residual_history(80.5, be.residuals_for("KMIA"))
buckets = candidate_buckets(distribution,anchor_f=80)
now = datetime(2026,9,5,1,0,tzinfo=timezone.utc)
stability = analyze([ForecastSnapshot(now-timedelta(seconds=1),80.5,80),ForecastSnapshot(now,80.5,80)],now=now)

frame = pd.read_parquet(ROOT / "data/features/tmax_24h_composite_training_v4.parquet")
model = joblib.load(ROOT / "artifacts/production_v4/xgb_residual_tmax.joblib")
availability = [col for col in frame if col.endswith("__availability")]
partial = frame[frame[availability].min(axis=1)<.9].copy()
partial["feature_completeness"] = feature_completeness(model,partial)
partial["core_guidance_pass"] = [has_complete_core_guidance(row) for row in partial.to_dict("records")]
partial[["station","target_date",*availability,"feature_completeness","core_guidance_pass"]].to_csv(OUT/"partial_profiles.csv",index=False)

result = {
    "probability_ceiling": float(ceilings.max_calibrated_probability.max()),
    "stations_passing_standard_probability_gate_possible": int((ceilings.max_calibrated_probability>=.58).sum()),
    "conditions": "Current v4 empirical residuals, shipped probability calibrator, default 2F buckets, supported day-ahead horizon, no same-day truncation.",
    "calibration_mass_example": {"station":"KMIA","point_f":80.5,"anchor_f":80,
        "raw_total":sum(x.probability for x in buckets),
        "calibrated_total":sum(cal.apply(x.probability) for x in buckets)},
    "one_second_stability_example":{"hours_observed":stability.hours_observed,
        "stability_score":stability.stability_score,"change_6h_f":stability.change_6h_f},
    "partial_profile_rows":len(partial),
    "partial_profiles_passing_existing_feature_and_core_gates":int(((partial.feature_completeness>=.85)&partial.core_guidance_pass).sum()),
    "profile_minimum_availability":frame[availability].min().to_dict(),
    "label_attributes":frame.tmax_attributes.value_counts().to_dict(),
}
heldout = pd.read_parquet(ROOT / "artifacts/backtest_v4/rolling_predictions.parquet")
hrrr = heldout["ncep_hrrr_conus__tmax_f"]
gfs = heldout["ncep_gfs_seamless__tmax_f"]
raw_samples = []
raw_paths = sorted((ROOT / "data/raw/open_meteo_previous_runs_v4").glob("*.json"))
for path in (raw_paths[0], raw_paths[len(raw_paths)//2], raw_paths[-2]):
    payload = json.loads(path.read_text(encoding="utf-8"))
    first_location = payload[0] if isinstance(payload,list) else payload
    hourly = first_location["hourly"]
    sample = {"file":str(path),"location_index":0,"variables":{}}
    for variable in ("temperature_2m","cloud_cover","wind_speed_10m"):
        a = hourly[variable+"_previous_day1_ncep_hrrr_conus"]
        b = hourly[variable+"_previous_day1_ncep_gfs_seamless"]
        pairs = [(x,y) for x,y in zip(a,b,strict=True) if x is not None and y is not None]
        sample["variables"][variable] = {"finite_pairs":len(pairs),"identical_pairs":sum(x==y for x,y in pairs)}
    raw_samples.append(sample)
result["source_duplication"] = {"same_high_rows":int((hrrr==gfs).sum()),"rows":len(heldout),
    "same_high_pct":float(100*(hrrr==gfs).mean()),
    "error_correlation":float((hrrr-heldout.tmax_f).corr(gfs-heldout.tmax_f)),
    "raw_samples":raw_samples,
    "interpretation":"Proven duplication in the saved inputs; upstream aliasing, provider fallback, or collection root cause has not been established."}
(OUT/"contract_reproductions.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
