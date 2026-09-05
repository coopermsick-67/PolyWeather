"""Recompute the v4 audit from frozen prediction rows; never calls live APIs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from polyweather.stations import STATIONS
from polyweather.bet_evidence import bucket_for
from polyweather.betfilter.results import settled_in_bucket

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "predictive_audit_2026-09-04"
OUT.mkdir(parents=True, exist_ok=True)
path = ROOT / "artifacts/backtest_v4/rolling_predictions.parquet"
df = pd.read_parquet(path)
df["target_date"] = pd.to_datetime(df["target_date"])
assert not df.duplicated(["station", "target_date"]).any()
assert set(df.station) == set(STATIONS)
assert df[["tmax_f", "xgb_prediction_f", "nbm_baseline_f"]].notna().all().all()
df["error"] = df.xgb_prediction_f - df.tmax_f
df["abs_error"] = df.error.abs()
df["within2"] = df.abs_error <= 2
df["within1"] = df.abs_error <= 1
buckets = [bucket_for(x) for x in df.xgb_prediction_f]
df["bucket_hit"] = [settled_in_bucket(a, *b) for a, b in zip(df.tmax_f, buckets)]
df["interval_hit"] = (df.tmax_f >= df.interval_lower_f) & (df.tmax_f <= df.interval_upper_f)
df["interval_width"] = df.interval_upper_f - df.interval_lower_f

def metrics(g):
    return {"n": len(g), "within2_hits": int(g.within2.sum()), "within2_pct": 100*g.within2.mean(),
            "within1_pct":100*g.within1.mean(), "bucket_hits":int(g.bucket_hit.sum()),
            "bucket_pct":100*g.bucket_hit.mean(), "mae_f":g.abs_error.mean(),
            "rmse_f":np.sqrt(np.mean(g.error**2)), "bias_f":g.error.mean(),
            "p90_error_f":g.abs_error.quantile(.9), "max_error_f":g.abs_error.max(),
            "over5_count":int((g.abs_error > 5).sum()),
            "interval_coverage_pct":100*g.interval_hit.mean(), "interval_width_f":g.interval_width.mean(),
            "nbm_mae_f":(g.nbm_baseline_f-g.tmax_f).abs().mean()}

station = pd.DataFrame([{"station":s,"city":STATIONS[s].display_name, **metrics(g)} for s,g in df.groupby("station")])
station.to_csv(OUT / "city_accuracy.csv", index=False)
monthly = pd.DataFrame([{"month":str(m),**metrics(g)} for m,g in df.groupby(df.target_date.dt.to_period("M"))])
monthly.to_csv(OUT / "monthly_accuracy.csv", index=False)
df.nlargest(30,"abs_error")[["station","target_date","tmax_f","xgb_prediction_f","nbm_baseline_f","error"]].to_csv(OUT / "largest_errors.csv", index=False)

# Resample whole dates; also use circular seven-day blocks to expose serial dependence.
daily = df.groupby("target_date")[["within2","bucket_hit"]].agg(["sum","count"])
rng = np.random.default_rng(20260904)
ci = {}
for block in (1,7):
    starts = rng.integers(0,len(daily),size=(4000,int(np.ceil(len(daily)/block))))
    ix = ((starts[:,:,None] + np.arange(block)) % len(daily)).reshape(4000,-1)[:,:len(daily)]
    for key in ("within2","bucket_hit"):
        values = daily[key]["sum"].to_numpy()[ix].sum(axis=1)/daily[key]["count"].to_numpy()[ix].sum(axis=1)
        ci[f"{key}_{block}day_block_95ci_pct"] = (100*np.quantile(values,[.025,.975])).tolist()

baselines = {}
for col in ("nbm_baseline_f","ncep_hrrr_conus__tmax_f","ncep_gfs_seamless__tmax_f","ridge_prediction_f","blend_prediction_f"):
    err=(df[col]-df.tmax_f).abs()
    hits=[settled_in_bucket(a,*bucket_for(p)) for a,p in zip(df.tmax_f,df[col])]
    baselines[col]={"mae_f":err.mean(),"within2_pct":100*(err<=2).mean(),"bucket_pct":100*np.mean(hits)}
summary={"source":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
         "start":str(df.target_date.min().date()),"end":str(df.target_date.max().date()),
         "dates":df.target_date.nunique(),"stations":df.station.nunique(),"duplicate_station_dates":0,
         "pooled":metrics(df),"equal_city_within2_pct":station.within2_pct.mean(),
         "equal_city_bucket_pct":station.bucket_pct.mean(),"bootstrap":ci,"baselines":baselines,
         "folds":df.fold_start.nunique(),"columns":list(df.columns),
         "threshold_abs_error_85pct_f":df.abs_error.quantile(.85),
         "recent_60_days":metrics(df[df.target_date >= df.target_date.max()-pd.Timedelta(days=59)])}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k != "columns"},indent=2,default=str))
print(station[["city","station","n","within2_pct","bucket_pct","mae_f","bias_f"]].to_string(index=False))
print(monthly[["month","n","within2_pct","bucket_pct","mae_f"]].to_string(index=False))
