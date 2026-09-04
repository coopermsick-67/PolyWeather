"""Did the filter actually help? Measured, not assumed.

Two questions this answers that a win rate alone cannot:

1. Would the rejected markets have lost? A filter that rejects markets at
   the same rate they would have won is destroying opportunity, not risk.
2. Where should the thresholds actually sit? Sweeping them against history
   is the only honest way to choose; picking 70% because it sounds
   disciplined is the same mistake as picking it because it looked good on
   one chart.

Every function here requires that rejected markets were logged too. If only
placed bets are stored, selection bias makes all of this meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("station", "target_date", "bucket_probability", "quality_score", "settled_in_bucket")


@dataclass(frozen=True)
class FilterEffectiveness:
    evaluated: int
    recommended: int
    coverage_rate: float
    recommended_wins: int
    recommended_win_rate: float | None
    unfiltered_wins: int
    unfiltered_win_rate: float | None
    lift_percentage_points: float | None
    rejected: int
    rejected_would_have_won: int
    rejected_would_have_lost: int
    avoided_loss_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluated": self.evaluated,
            "recommended": self.recommended,
            "coverageRate": round(self.coverage_rate, 4),
            "recommendedWins": self.recommended_wins,
            "recommendedWinRate": None if self.recommended_win_rate is None else round(self.recommended_win_rate, 4),
            "unfilteredWinRate": None if self.unfiltered_win_rate is None else round(self.unfiltered_win_rate, 4),
            "liftPercentagePoints": None if self.lift_percentage_points is None else round(self.lift_percentage_points, 2),
            "rejected": self.rejected,
            "rejectedWouldHaveWon": self.rejected_would_have_won,
            "rejectedWouldHaveLost": self.rejected_would_have_lost,
            "avoidedLossRate": None if self.avoided_loss_rate is None else round(self.avoided_loss_rate, 4),
        }


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Decision log is missing required columns: {', '.join(missing)}")
    resolved = frame.loc[frame["settled_in_bucket"].notna()].copy()
    resolved["settled_in_bucket"] = resolved["settled_in_bucket"].astype(bool)
    return resolved


def effectiveness(
    decision_log: pd.DataFrame,
    minimum_probability: float,
    minimum_quality_score: float,
) -> FilterEffectiveness:
    """Compare the filter's picks against betting every model favorite.

    ``unfiltered`` is the honest control: it is what the app did before,
    taking the top bucket in every city. Beating it is the only thing that
    justifies the filter's existence.
    """
    resolved = _validate(decision_log)
    if resolved.empty:
        return FilterEffectiveness(0, 0, 0.0, 0, None, 0, None, None, 0, 0, 0, None)
    recommended_mask = (
        (resolved["bucket_probability"] >= minimum_probability)
        & (resolved["quality_score"] >= minimum_quality_score)
    )
    recommended = resolved.loc[recommended_mask]
    rejected = resolved.loc[~recommended_mask]
    unfiltered_wins = int(resolved["settled_in_bucket"].sum())
    recommended_wins = int(recommended["settled_in_bucket"].sum())
    rejected_wins = int(rejected["settled_in_bucket"].sum())
    recommended_rate = recommended_wins / len(recommended) if len(recommended) else None
    unfiltered_rate = unfiltered_wins / len(resolved)
    return FilterEffectiveness(
        evaluated=len(resolved),
        recommended=len(recommended),
        coverage_rate=len(recommended) / len(resolved),
        recommended_wins=recommended_wins,
        recommended_win_rate=recommended_rate,
        unfiltered_wins=unfiltered_wins,
        unfiltered_win_rate=unfiltered_rate,
        lift_percentage_points=(
            100 * (recommended_rate - unfiltered_rate) if recommended_rate is not None else None
        ),
        rejected=len(rejected),
        rejected_would_have_won=rejected_wins,
        rejected_would_have_lost=len(rejected) - rejected_wins,
        avoided_loss_rate=(len(rejected) - rejected_wins) / len(rejected) if len(rejected) else None,
    )


def sweep_thresholds(
    decision_log: pd.DataFrame,
    probability_grid: tuple[float, ...] = (0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80),
    quality_grid: tuple[float, ...] = (0.0, 65.0, 70.0, 75.0, 80.0, 82.0, 85.0, 88.0, 90.0),
    minimum_sample: int = 10,
) -> pd.DataFrame:
    """Grid-search thresholds against resolved history.

    Rows below ``minimum_sample`` are still returned but flagged: a 100% win
    rate from four bets is the single most misleading number this table can
    produce, and hiding it entirely invites someone to re-derive it later.
    """
    resolved = _validate(decision_log)
    rows: list[dict[str, object]] = []
    for probability in probability_grid:
        for quality in quality_grid:
            result = effectiveness(resolved, probability, quality)
            rows.append({
                "minimum_probability": probability,
                "minimum_quality_score": quality,
                "bets": result.recommended,
                "wins": result.recommended_wins,
                "win_rate": result.recommended_win_rate,
                "coverage": result.coverage_rate,
                "lift_pp": result.lift_percentage_points,
                "sufficient_sample": result.recommended >= minimum_sample,
            })
    return pd.DataFrame(rows)


def calibration_table(
    decision_log: pd.DataFrame,
    bins: tuple[float, ...] = (0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01),
) -> pd.DataFrame:
    """Do the stated probabilities happen at their stated rate?

    If markets labeled 70% land 55% of the time, every downstream number is
    overconfident and the filter's thresholds are measuring the wrong scale.
    """
    resolved = _validate(decision_log)
    if resolved.empty:
        return pd.DataFrame(columns=["bin_lower", "bin_upper", "n", "mean_predicted", "observed_rate", "gap"])
    resolved = resolved.assign(
        bin=pd.cut(resolved["bucket_probability"], bins=list(bins), right=False)
    )
    rows: list[dict[str, object]] = []
    for interval, group in resolved.groupby("bin", observed=True, sort=True):
        if group.empty:
            continue
        predicted = float(group["bucket_probability"].mean())
        observed = float(group["settled_in_bucket"].mean())
        rows.append({
            "bin_lower": float(interval.left),
            "bin_upper": float(interval.right),
            "n": int(len(group)),
            "mean_predicted": round(predicted, 4),
            "observed_rate": round(observed, 4),
            "gap": round(observed - predicted, 4),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(decision_log: pd.DataFrame) -> float:
    """Sample-weighted mean gap between stated and realized probability."""
    table = calibration_table(decision_log)
    if table.empty:
        return float("nan")
    weights = table["n"].to_numpy(float)
    return float(np.average(np.abs(table["gap"].to_numpy(float)), weights=weights))


def brier_score(decision_log: pd.DataFrame) -> float:
    """Mean squared error of the probability forecasts. Lower is better."""
    resolved = _validate(decision_log)
    if resolved.empty:
        return float("nan")
    predicted = resolved["bucket_probability"].to_numpy(float)
    outcome = resolved["settled_in_bucket"].to_numpy(float)
    return float(np.mean((predicted - outcome) ** 2))


def rejection_reason_counts(decision_log: pd.DataFrame, reason_column: str = "primary_reason") -> pd.DataFrame:
    """How often each reason fired, and whether rejecting for it was right."""
    if reason_column not in decision_log:
        raise ValueError(f"Decision log has no '{reason_column}' column.")
    resolved = _validate(decision_log)
    rejected = resolved.loc[resolved[reason_column].notna()]
    if rejected.empty:
        return pd.DataFrame(columns=[reason_column, "count", "would_have_won", "avoided_loss_rate"])
    grouped = rejected.groupby(reason_column).agg(
        count=("settled_in_bucket", "size"),
        would_have_won=("settled_in_bucket", "sum"),
    ).reset_index()
    grouped["avoided_loss_rate"] = 1 - grouped["would_have_won"] / grouped["count"]
    return grouped.sort_values("count", ascending=False).reset_index(drop=True)
