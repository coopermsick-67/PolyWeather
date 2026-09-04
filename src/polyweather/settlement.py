"""Versioned settlement-contract registry and bucket-hit-rate scoring.

This module ships with **zero populated real `MarketContract` records** by
design. Populating an entry requires a real, sourced settlement contract
(bucket widths, rounding rule, exact station/date resolution rule) that has
actually been verified against a real prediction-market platform. Fabricating
a plausible-looking contract would defeat the entire purpose of this
registry, so `load_market_contracts` and `MarketContract.__post_init__`
reject anything unverified or malformed rather than guessing.

Nothing here changes `dashboard_payload.py`'s or `create-worker.mjs`'s live
`NO_BET` default, and nothing here is wired into the live dashboard's
`dataQualityStatus` output. This is backtesting/research plumbing only -- the
repo owner decides separately, with real verified market data, whether/when
to change anything live.

Two very different kinds of numbers live in this module and must never be
confused:

* :func:`bucket_hit_rate` -- a real hit-rate/calibration number, but it can
  ONLY be computed from a verified `MarketContract` (``verified_at`` and
  ``verified_by`` set). It raises if given anything else.
* :func:`illustrative_bucket_hit_rate` -- an exploratory, UNVERIFIED number
  computed against :data:`EXAMPLE_BUCKET_WIDTHS` (a plausible-but-unsourced
  2 degF research default) or any other bucket width the caller supplies. It
  is not tied to any real settlement contract. Every field name in its
  output is prefixed ``illustrative_`` so it can never be mistaken for a
  verified, contract-backed number.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from .markets import normal_bracket_probabilities
from .stations import require_station

# Purely illustrative, 2 degF-wide bucket definition for exploring what a
# bucket-based hit rate WOULD look like under a plausible-but-UNVERIFIED
# bucket width. This is NOT a real settlement contract for any real
# prediction-market platform -- it exists only so `illustrative_bucket_hit_rate`
# has a usable default when no verified `MarketContract` exists yet (which,
# per this module's ground rules, is expected to be the common case).
EXAMPLE_BUCKET_WIDTHS: tuple[tuple[int, int], ...] = tuple((low, low + 1) for low in range(-20, 121, 2))

# Rounding rules this module knows how to apply. Deliberately explicit and
# closed -- an unrecognized rounding_rule string is rejected rather than
# guessed at scoring time.
_ROUND_HALF_UP = "round half up"
_ROUND_HALF_TO_EVEN = "round half to even"
SUPPORTED_ROUNDING_RULES: tuple[str, ...] = (_ROUND_HALF_UP, _ROUND_HALF_TO_EVEN)

REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = ("station", "target_date", "xgb_prediction_f", "tmax_f", "p10_f", "p90_f")

# Standard-normal quantile at the 90th percentile (and, by symmetry, minus
# this value at the 10th percentile). Used to imply a standard deviation from
# a model's own reported p10_f/p90_f split-conformal interval.
_Z_90 = 1.2815515655446004


@dataclass(frozen=True)
class MarketContract:
    """A versioned, sourced prediction-market settlement contract.

    A contract with ``verified_at is None`` (or any malformed field) must
    never be usable for real hit-rate scoring. `__post_init__` enforces that
    at construction time -- there is no way to build an "unverified"
    `MarketContract` instance -- and :func:`bucket_hit_rate` re-checks
    `is_verified` defensively at scoring time as well.
    """

    station_id: str
    provider: str
    settlement_source: str
    bucket_definition: tuple[tuple[int, int], ...]
    rounding_rule: str
    contract_version: str
    verified_at: date | None
    verified_by: str | None

    def __post_init__(self) -> None:
        station = require_station(self.station_id)
        if self.settlement_source not in station.source_priority:
            raise ValueError(
                f"settlement_source {self.settlement_source!r} for {self.station_id} must reference one of "
                f"the station's own source_priority entries {station.source_priority!r}, not an invented source."
            )
        if not self.bucket_definition:
            raise ValueError("bucket_definition must contain at least one (low, high) bucket.")
        for low, high in self.bucket_definition:
            if not isinstance(low, int) or not isinstance(high, int):
                raise ValueError(f"Bucket bounds must be integer degrees, got ({low!r}, {high!r}).")
            if low > high:
                raise ValueError(f"Bucket ({low}, {high}) has low > high.")
        if self.rounding_rule not in SUPPORTED_ROUNDING_RULES:
            raise ValueError(f"rounding_rule must be one of {SUPPORTED_ROUNDING_RULES!r}, not a guessed value: {self.rounding_rule!r}.")
        if not self.contract_version or not self.contract_version.strip():
            raise ValueError("contract_version must be a non-empty version string.")
        if self.verified_at is None or not self.verified_by or not str(self.verified_by).strip():
            raise ValueError(
                "MarketContract requires both verified_at and verified_by to be set. An unverified or "
                "unsourced settlement contract must be left out of the registry entirely, never guessed."
            )

    @property
    def is_verified(self) -> bool:
        """True only when both verification fields are populated.

        Given `__post_init__` above, a constructed `MarketContract` is always
        verified -- this property exists so call sites (and tests) can assert
        the guard explicitly rather than relying on construction never having
        been bypassed.
        """
        return self.verified_at is not None and bool(self.verified_by) and bool(str(self.verified_by).strip())

    @property
    def station(self):
        return require_station(self.station_id)


def load_market_contracts(path: str | Path) -> list[MarketContract]:
    """Load settlement contracts from a JSON file.

    Same pattern as :func:`polyweather.stations.load_station_registry`: a
    JSON list of records, each turned into a validated dataclass, with any
    malformed or unverified record raising immediately rather than being
    silently coerced into something plausible-looking. It is expected and
    fine for this file to contain an empty list -- there is no requirement
    to populate any real contracts before this plumbing is useful.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Market contract registry file must contain a JSON list.")
    loaded: list[MarketContract] = []
    seen_versions: set[tuple[str, str]] = set()
    for record in records:
        try:
            verified_at_raw = record.get("verified_at")
            contract = MarketContract(
                station_id=str(record["station_id"]).upper(),
                provider=str(record["provider"]),
                settlement_source=str(record["settlement_source"]),
                bucket_definition=tuple((int(low), int(high)) for low, high in record["bucket_definition"]),
                rounding_rule=str(record["rounding_rule"]),
                contract_version=str(record["contract_version"]),
                verified_at=date.fromisoformat(str(verified_at_raw)) if verified_at_raw else None,
                verified_by=str(record["verified_by"]) if record.get("verified_by") else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid or unverified market contract entry, rejected rather than guessed: {record!r}") from exc
        key = (contract.station_id, contract.contract_version)
        if key in seen_versions:
            raise ValueError(f"Duplicate market contract station/version: {key}.")
        seen_versions.add(key)
        loaded.append(contract)
    return loaded


def _round_temperature(value: float, rounding_rule: str) -> int:
    """Apply an explicit, documented rounding rule -- never an assumed one."""
    if rounding_rule == _ROUND_HALF_TO_EVEN:
        # Banker's rounding: ties round to the nearest even integer.
        return int(round(value))
    if rounding_rule == _ROUND_HALF_UP:
        return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)
    raise ValueError(f"Unsupported rounding_rule {rounding_rule!r}. Must be one of {SUPPORTED_ROUNDING_RULES!r}.")


def _bucket_key(low: int, high: int) -> str:
    """Match `normal_bracket_probabilities`'s own bracket-key format exactly."""
    return f"{low:g}-{high:g}"


def _find_bucket(value: int, buckets: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    for low, high in buckets:
        if low <= value <= high:
            return (low, high)
    return None


def _implied_std_from_interval(p10: float, p90: float) -> float:
    """Back out an implied normal standard deviation from a p10/p90 split."""
    return (p90 - p10) / (2 * _Z_90)


def _score_predictions(predictions: pd.DataFrame, bucket_definition: Sequence[tuple[int, int]], rounding_rule: str) -> pd.DataFrame:
    """Row-level bucket scoring shared by the verified and illustrative paths.

    For each row: which bucket the model's point prediction falls into
    (after applying `rounding_rule`), whether that same bucket actually
    contained the observed `tmax_f` (also rounded), and the model's own
    claimed probability mass on that winning bucket via
    `normal_bracket_probabilities`, using a standard deviation implied by the
    model's own reported p10_f/p90_f interval. Rows with a degenerate
    (non-positive) implied interval are skipped rather than guessed.
    """
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")
    bucket_definition = tuple(bucket_definition)
    rows: list[dict[str, object]] = []
    for row in predictions.itertuples(index=False):
        mean = float(getattr(row, "xgb_prediction_f"))
        actual = float(getattr(row, "tmax_f"))
        p10 = float(getattr(row, "p10_f"))
        p90 = float(getattr(row, "p90_f"))
        if pd.isna(mean) or pd.isna(actual) or pd.isna(p10) or pd.isna(p90):
            continue
        standard_deviation = _implied_std_from_interval(p10, p90)
        if standard_deviation <= 0:
            continue
        predicted_value = _round_temperature(mean, rounding_rule)
        actual_value = _round_temperature(actual, rounding_rule)
        predicted_bucket = _find_bucket(predicted_value, bucket_definition)
        actual_bucket = _find_bucket(actual_value, bucket_definition)
        claimed_probability = 0.0
        predicted_bucket_key = None
        if predicted_bucket is not None:
            predicted_bucket_key = _bucket_key(*predicted_bucket)
            probabilities = normal_bracket_probabilities(mean, standard_deviation, bucket_definition)
            claimed_probability = probabilities[predicted_bucket_key]
        rows.append(
            {
                "station": getattr(row, "station"),
                "target_date": getattr(row, "target_date"),
                "predicted_bucket": predicted_bucket_key,
                "hit": bool(predicted_bucket is not None and predicted_bucket == actual_bucket),
                "claimed_probability": claimed_probability,
            }
        )
    return pd.DataFrame(rows, columns=["station", "target_date", "predicted_bucket", "hit", "claimed_probability"])


def _calibration_table(scored: pd.DataFrame, bins: int = 10) -> list[dict[str, object]]:
    """Bin claimed probability vs. observed hit frequency -- a real calibration check."""
    if scored.empty:
        return []
    working = scored.copy()
    working["probability_bin"] = pd.cut(working["claimed_probability"], bins=bins, include_lowest=True)
    grouped = (
        working.groupby("probability_bin", observed=True)
        .agg(mean_claimed_probability=("claimed_probability", "mean"), observed_hit_frequency=("hit", "mean"), n=("hit", "size"))
        .reset_index()
    )
    grouped["probability_bin"] = grouped["probability_bin"].astype(str)
    return grouped.to_dict(orient="records")


def bucket_hit_rate(predictions: pd.DataFrame, contract: MarketContract) -> dict[str, object]:
    """Score a REAL bucket-hit-rate and calibration check for a verified contract.

    `predictions` follows the shape of `artifacts/backtest_v4/rolling_predictions.parquet`
    (columns `station`, `target_date`, `xgb_prediction_f`, `tmax_f`, `p10_f`, `p90_f`).
    Raises `ValueError` if `contract` is not verified -- there is no code
    path in this module that produces a `hit_rate` number from an unverified
    contract.
    """
    if not contract.is_verified:
        raise ValueError("bucket_hit_rate requires a verified MarketContract (verified_at and verified_by must be set).")
    station_rows = predictions.loc[predictions["station"].astype(str).str.upper() == contract.station_id.upper()]
    scored = _score_predictions(station_rows, contract.bucket_definition, contract.rounding_rule)
    return {
        "station_id": contract.station_id,
        "provider": contract.provider,
        "settlement_source": contract.settlement_source,
        "contract_version": contract.contract_version,
        "verified_at": contract.verified_at.isoformat(),
        "verified_by": contract.verified_by,
        "rounding_rule": contract.rounding_rule,
        "n_predictions": int(len(scored)),
        "bucket_hit_rate": float(scored["hit"].mean()) if not scored.empty else None,
        "mean_claimed_probability_on_winning_bucket": float(scored["claimed_probability"].mean()) if not scored.empty else None,
        "calibration_table": _calibration_table(scored),
    }


def illustrative_bucket_hit_rate(
    predictions: pd.DataFrame,
    *,
    station_id: str | None = None,
    bucket_widths: Sequence[tuple[int, int]] = EXAMPLE_BUCKET_WIDTHS,
    rounding_rule: str = _ROUND_HALF_UP,
) -> dict[str, object]:
    """Illustrative, UNVERIFIED exploration of bucket hit-rate/calibration.

    THIS IS NOT A REAL, CONTRACT-VERIFIED NUMBER. No real prediction-market
    settlement contract backs `bucket_widths` (default: `EXAMPLE_BUCKET_WIDTHS`,
    a plausible-but-unsourced 2 degF research default) -- it exists purely to
    explore what a bucket-based hit rate WOULD look like under a plausible
    bucket width, given no verified contract exists yet. Every output field
    is prefixed `illustrative_` so it can never be confused with
    `bucket_hit_rate`'s verified-contract output.
    """
    working = predictions
    if station_id is not None:
        working = predictions.loc[predictions["station"].astype(str).str.upper() == station_id.upper()]
    bucket_widths = tuple(bucket_widths)
    scored = _score_predictions(working, bucket_widths, rounding_rule)
    return {
        "station_id": station_id,
        "illustrative_bucket_width_count": len(bucket_widths),
        "illustrative_rounding_rule": rounding_rule,
        "illustrative_n_predictions": int(len(scored)),
        "illustrative_bucket_hit_rate": float(scored["hit"].mean()) if not scored.empty else None,
        "illustrative_mean_claimed_probability_on_winning_bucket": float(scored["claimed_probability"].mean()) if not scored.empty else None,
        "illustrative_calibration_table": _calibration_table(scored),
        "warning": "ILLUSTRATIVE / UNVERIFIED: no real settlement contract backs this number. Do not treat as a real hit rate.",
    }


def illustrative_report_by_station(
    predictions: pd.DataFrame,
    *,
    bucket_widths: Sequence[tuple[int, int]] = EXAMPLE_BUCKET_WIDTHS,
    rounding_rule: str = _ROUND_HALF_UP,
) -> list[dict[str, object]]:
    """Station-by-station illustrative bucket-hit-rate/calibration report."""
    stations = sorted(str(value) for value in predictions["station"].dropna().unique())
    return [
        illustrative_bucket_hit_rate(predictions, station_id=station, bucket_widths=bucket_widths, rounding_rule=rounding_rule)
        for station in stations
    ]
