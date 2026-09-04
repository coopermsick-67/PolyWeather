"""Static QA figures and a concise technical model card."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4a5568",
            "axes.labelcolor": "#172033",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
        }
    )


def build_backtest_charts(backtest_dir: str | Path, report_dir: str | Path) -> list[Path]:
    """Render reviewed static figures used by the technical handoff."""
    _style()
    source = Path(backtest_dir)
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    overall = pd.read_csv(source / "overall_metrics.csv")
    station = pd.read_csv(source / "station_metrics.csv")
    predictions = pd.read_parquet(source / "rolling_predictions.parquet")
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    display = overall.sort_values("mae_f", ascending=False)
    bars = ax.barh(display["model"], display["mae_f"], color="#2d6cdf", edgecolor="#1e3a5f")
    ax.set_title("Held-out daily Tmax MAE by model")
    ax.set_xlabel("Mean absolute error (°F)")
    ax.set_xlim(0, max(display["mae_f"].max() * 1.2, 1))
    for bar, value in zip(bars, display["mae_f"], strict=True):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.2f}°F", va="center", fontsize=10)
    output = destination / "heldout_mae_by_model.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)

    focus = station.loc[station["model"].isin(["NBM", "XGBoost residual"])].copy()
    stations = sorted(focus["station"].unique())
    positions = np.arange(len(stations))
    width = 0.36
    nbm = focus.loc[focus["model"] == "NBM"].set_index("station").reindex(stations)
    xgb = focus.loc[focus["model"] == "XGBoost residual"].set_index("station").reindex(stations)
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    ax.bar(positions - width / 2, nbm["mae_f"], width, label="NBM baseline", color="#96a3b8", edgecolor="#4a5568")
    ax.bar(positions + width / 2, xgb["mae_f"], width, label="XGBoost residual", color="#2d6cdf", edgecolor="#1e3a5f")
    ax.set_xticks(positions, stations)
    ax.set_ylim(0, max(focus["mae_f"].max() * 1.2, 1))
    ax.set_xlabel("Station")
    ax.set_ylabel("Held-out MAE (°F)")
    ax.set_title("Held-out daily Tmax MAE by station")
    ax.legend(frameon=False)
    output = destination / "heldout_mae_by_station.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)

    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    daily = (
        predictions.assign(
            nbm_abs_error=(predictions["nbm_baseline_f"] - predictions["tmax_f"]).abs(),
            xgb_abs_error=(predictions["xgb_prediction_f"] - predictions["tmax_f"]).abs(),
        )
        .groupby("target_date")[["nbm_abs_error", "xgb_abs_error"]]
        .mean()
        .rolling(21, min_periods=7)
        .mean()
    )
    fig, ax = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    ax.plot(daily.index, daily["nbm_abs_error"], label="NBM baseline", color="#64748b", linewidth=1.8)
    ax.plot(daily.index, daily["xgb_abs_error"], label="XGBoost residual", color="#2d6cdf", linewidth=1.8)
    ax.set_title("21-day rolling held-out absolute error")
    ax.set_ylabel("Mean absolute error (°F)")
    ax.set_xlabel("Target date")
    ax.legend(frameon=False)
    output = destination / "rolling_heldout_error.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(output)
    return outputs


def make_model_card(backtest_dir: str | Path, output: str | Path) -> Path:
    """Write a transparent model card that explicitly bounds accuracy claims."""
    source = Path(backtest_dir)
    target = Path(output)
    overall = pd.read_csv(source / "overall_metrics.csv")
    station = pd.read_csv(source / "station_metrics.csv")
    bootstrap = pd.read_csv(source / "block_bootstrap_ci.csv")
    acceptance = json.loads((source / "acceptance.json").read_text(encoding="utf-8"))
    xgb = overall.loc[overall["model"] == "XGBoost residual"].iloc[0]
    nbm = overall.loc[overall["model"] == "NBM"].iloc[0]
    station_table = station.loc[station["model"].isin(["NBM", "XGBoost residual"])]
    pivot = station_table.pivot(index="station", columns="model", values=["n", "mae_f", "bias_f", "mae_skill_vs_nbm"])
    lines = [
        "# PolyWeather daily Tmax model card",
        "",
        "## Decision",
        "",
        f"**{acceptance['decision']}** - {acceptance['reason']}",
        "",
        "## What was predicted",
        "",
        "Official daily maximum temperature (TMAX, °F) at the 20 settlement stations declared in the model manifest. Labels are NOAA/NCEI daily-summaries records, not maxima reconstructed from rounded METAR observations.",
        "",
        "## Method",
        "",
        "The shadow candidate preserves the archived NCEP NBM hourly temperature curve as its baseline and uses gradient-boosted residual learning over NBM, HRRR, and GFS forecast profiles plus station ID and seasonal features. The backtest uses contiguous, calendar-ordered 31-day folds; no target date is scored twice, no random split or future reanalysis field is used.",
        "",
        "## Held-out result",
        "",
        f"Across all rolling held-out station-days, NBM MAE was **{nbm['mae_f']:.2f}°F** and the XGBoost residual model MAE was **{xgb['mae_f']:.2f}°F**. Relative MAE skill was **{xgb['mae_skill_vs_nbm']:.1%}**.",
        "",
        "| station | NBM MAE °F | XGBoost MAE °F | XGBoost bias °F | XGBoost skill vs NBM | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for station_name in sorted(station_table["station"].unique()):
        n = pivot.loc[station_name, ("n", "XGBoost residual")]
        nbm_mae = pivot.loc[station_name, ("mae_f", "NBM")]
        xgb_mae = pivot.loc[station_name, ("mae_f", "XGBoost residual")]
        bias = pivot.loc[station_name, ("bias_f", "XGBoost residual")]
        skill = pivot.loc[station_name, ("mae_skill_vs_nbm", "XGBoost residual")]
        lines.append(f"| {station_name} | {nbm_mae:.2f} | {xgb_mae:.2f} | {bias:+.2f} | {skill:.1%} | {int(n)} |")
    if not bootstrap.empty:
        xgb_ci = bootstrap.loc[bootstrap["model"] == "XGBoost residual"].iloc[0]
        lines.extend(
            [
                "",
                "A date-block bootstrap 95% interval for the XGBoost MAE is "
                f"**{xgb_ci['mae_ci_low']:.2f}-{xgb_ci['mae_ci_high']:.2f}°F**; its skill interval versus NBM is "
                f"**{xgb_ci['skill_ci_low']:.1%}-{xgb_ci['skill_ci_high']:.1%}**.",
            ]
        )
    lines.extend(
        [
            "",
            "## Critical caveats",
            "",
            "- This is a fixed-lead hourly reconstruction: each input hourly value was archived at a 24-hour lead, then aggregated to local-day Tmax. It is leakage-resistant, but it is not a reconstruction of one frozen once-per-day NWS forecast issuance.",
            "- The current model uses the public archived NCEP NBM/HRRR/GFS forecast feed as its numerical guidance source. It does not yet include a historical immutable NWS forecastGridData archive, raw NBM GRIB neighborhood features, real-time ASOS histories, radar, satellite, or SST.",
            "- The reported score is specific to the tested period, stations, target definition, lead convention, sources, and their model versions. It is not a promise for future daily highs; keep the model in shadow monitoring through seasonal transitions and upstream NWP changes.",
            "- Prediction intervals are asymmetric split-conformal, nominal 80% bands calibrated using prior dates only with conservative finite-sample order statistics. Monitor empirical coverage after deployment.",
            "",
            "## Reproducibility",
            "",
            "Artifacts include the feature table checksum, row-level rolling predictions, per-station metrics, date-block bootstrap confidence intervals, acceptance decision, serialized model, and run manifest.",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
