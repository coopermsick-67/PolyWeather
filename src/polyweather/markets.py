"""Typed market metadata, safe discovery, and prediction-quality gates.

No PrizePicks endpoint is queried here.  Public/approved adapters and manual
CSV/JSON input feed the same validation path, leaving ambiguous settlement
locations for a human to review instead of silently choosing a city centre.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from .stations import Station, require_station


class MarketType(StrEnum):
    DAILY_HIGH = "daily_high"
    DAILY_LOW = "daily_low"
    TEMPERATURE_BRACKET = "temperature_bracket"
    THRESHOLD_GTE = "threshold_gte"
    THRESHOLD_LTE = "threshold_lte"
    HOURLY_TEMPERATURE = "hourly_temperature"
    PRECIPITATION = "precipitation"
    SNOW = "snow"
    WIND = "wind"
    SEVERE_WEATHER = "severe_weather"


class QualityStatus(StrEnum):
    VERIFIED = "VERIFIED"
    PROVISIONAL = "PROVISIONAL"
    STALE_DATA = "STALE DATA"
    SOURCE_CONFLICT = "SOURCE CONFLICT"
    UNKNOWN_SETTLEMENT_RULE = "UNKNOWN SETTLEMENT RULE"
    NO_BET = "NO BET / INSUFFICIENT DATA"


@dataclass(frozen=True)
class MarketSpec:
    market_id: str
    station_id: str
    target_date: date
    market_type: MarketType
    settlement_source: str | None
    settlement_rules_url: str | None = None
    lower_f: float | None = None
    upper_f: float | None = None
    threshold_f: float | None = None
    close_time: datetime | None = None
    resolution_time: datetime | None = None
    hourly_observation_time: datetime | None = None

    def __post_init__(self) -> None:
        require_station(self.station_id)
        if self.market_type == MarketType.TEMPERATURE_BRACKET and (self.lower_f is None or self.upper_f is None or self.lower_f > self.upper_f):
            raise ValueError("A temperature bracket requires ordered lower_f and upper_f values.")
        if self.market_type in {MarketType.THRESHOLD_GTE, MarketType.THRESHOLD_LTE} and self.threshold_f is None:
            raise ValueError("A threshold market requires threshold_f.")
        if self.market_type == MarketType.HOURLY_TEMPERATURE and self.hourly_observation_time is None:
            raise ValueError("Hourly markets require an exact observation time and provider rules.")

    @property
    def station(self) -> Station:
        return require_station(self.station_id)

    @property
    def timezone(self) -> str:
        return self.station.timezone


@dataclass(frozen=True)
class PredictionRecord:
    market_id: str
    market_location: str
    station_id: str
    target_date: date
    timezone: str
    market_type: MarketType
    settlement_source: str | None
    settlement_rules_url: str | None
    current_observed_temperature: float | None
    observed_daily_high: float | None
    observed_daily_low: float | None
    model_mean: float | None
    model_median: float | None
    p10: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p90: float | None
    bracket_probabilities: dict[str, float]
    confidence: float
    data_freshness: float
    source_agreement: float
    generated_at: datetime
    model_version: str
    status: QualityStatus
    reason_codes: tuple[str, ...] = ()


def normal_bracket_probabilities(mean: float, standard_deviation: float, brackets: Iterable[tuple[float, float]]) -> dict[str, float]:
    """Return inclusive integer-degree-bin mass from a normal approximation."""
    from math import erf, sqrt

    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive.")
    cdf = lambda value: 0.5 * (1.0 + erf((value - mean) / (standard_deviation * sqrt(2))))
    return {f"{low:g}-{high:g}": max(0.0, min(1.0, cdf(high + 0.5) - cdf(low - 0.5))) for low, high in brackets}


def quality_gate(
    *, station_known: bool, rules_available: bool, freshness: float, agreement: float,
    interval_width_f: float | None, stale_below: float = 0.55, agreement_below: float = 0.45,
    max_interval_width_f: float = 8.0,
) -> QualityStatus:
    if not station_known or not rules_available:
        return QualityStatus.UNKNOWN_SETTLEMENT_RULE
    if freshness < stale_below:
        return QualityStatus.STALE_DATA
    if agreement < agreement_below:
        return QualityStatus.SOURCE_CONFLICT
    if interval_width_f is None or interval_width_f > max_interval_width_f:
        return QualityStatus.NO_BET
    return QualityStatus.PROVISIONAL


def _parse_temperature_terms(title: str) -> tuple[MarketType, float | None, float | None, float | None]:
    normalized = title.casefold()
    bracket = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(-?\d+(?:\.\d+)?)\s*°?f?", normalized)
    if bracket:
        return MarketType.TEMPERATURE_BRACKET, float(bracket.group(1)), float(bracket.group(2)), None
    threshold = re.search(r"(?:at least|over|above|>=|greater than or equal to)\s*(-?\d+(?:\.\d+)?)", normalized)
    if threshold:
        return MarketType.THRESHOLD_GTE, None, None, float(threshold.group(1))
    threshold = re.search(r"(?:at most|under|below|<=|less than or equal to)\s*(-?\d+(?:\.\d+)?)", normalized)
    if threshold:
        return MarketType.THRESHOLD_LTE, None, None, float(threshold.group(1))
    return (MarketType.DAILY_LOW if "low" in normalized else MarketType.DAILY_HIGH), None, None, None


def parse_manual_market(record: dict[str, object]) -> MarketSpec:
    """Parse one approved/manual record and reject any ambiguous city mapping."""
    station_hint = str(record.get("stationId") or record.get("station_id") or record.get("location") or "")
    try:
        station = require_station(station_hint)
    except ValueError as exc:
        raise ValueError(f"Manual-review required: cannot map market location {station_hint!r} to one settlement station.") from exc
    title = str(record.get("title") or "")
    market_type, lower, upper, threshold = _parse_temperature_terms(title)
    if record.get("market_type"):
        market_type = MarketType(str(record["market_type"]))
    target = date.fromisoformat(str(record["targetDate"] or record["target_date"]))
    return MarketSpec(
        market_id=str(record.get("marketId") or record.get("market_id") or f"manual-{station.slug}-{target.isoformat()}"),
        station_id=station.icao, target_date=target, market_type=market_type,
        settlement_source=str(record.get("settlementSource") or record.get("settlement_source") or "manual rules pending"),
        settlement_rules_url=str(record["settlementRulesUrl"]) if record.get("settlementRulesUrl") else None,
        lower_f=float(record.get("lower_f", lower)) if record.get("lower_f", lower) is not None else None,
        upper_f=float(record.get("upper_f", upper)) if record.get("upper_f", upper) is not None else None,
        threshold_f=float(record.get("threshold_f", threshold)) if record.get("threshold_f", threshold) is not None else None,
    )


MAX_IMPORT_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMPORT_RECORDS = 20_000


def import_markets(path: str | Path) -> tuple[list[MarketSpec], list[dict[str, str]]]:
    """Import JSON/CSV market records, retaining rejected rows for review."""
    source = Path(path)
    file_size = source.stat().st_size
    if file_size > MAX_IMPORT_FILE_BYTES:
        raise ValueError(f"Market import file is {file_size} bytes, exceeding the {MAX_IMPORT_FILE_BYTES}-byte limit.")
    if source.suffix.casefold() == ".json":
        records = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.casefold() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    else:
        raise ValueError("Only JSON and CSV market imports are supported.")
    if not isinstance(records, list):
        raise ValueError("Market import must contain a list of records.")
    if len(records) > MAX_IMPORT_RECORDS:
        raise ValueError(f"Market import contains {len(records)} records, exceeding the {MAX_IMPORT_RECORDS}-record limit.")
    accepted: list[MarketSpec] = []
    rejected: list[dict[str, str]] = []
    for raw in records:
        try:
            accepted.append(parse_manual_market(raw))
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"record": json.dumps(raw, default=str), "reason": str(exc)})
    return accepted, rejected


def serialize_prediction(record: PredictionRecord) -> dict[str, object]:
    value = asdict(record)
    value["target_date"] = record.target_date.isoformat()
    value["generated_at"] = record.generated_at.isoformat()
    value["market_type"] = record.market_type.value
    value["status"] = record.status.value
    return value
