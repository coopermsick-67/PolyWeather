"""Small official-NWS client with bounded retries, caching, and provenance."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from threading import Lock
from typing import Any

import requests

from .stations import Station


NWS_BASE_URL = "https://api.weather.gov"
MAX_CACHE_ENTRIES = 256
# NWS asks API consumers to identify themselves with a real contact so they
# can reach the operator about problem traffic; https://www.weather.gov/documentation/services-web-api
# Override via NWS_USER_AGENT in any deployment that has a real contact
# point, rather than shipping the placeholder to production indefinitely.
DEFAULT_USER_AGENT = os.environ.get("NWS_USER_AGENT", "PolyWeather/0.2 (set NWS_USER_AGENT to a real contact)")


@dataclass(frozen=True)
class SourceSnapshot:
    source: str
    retrieved_at: datetime
    source_time: datetime | None
    stale: bool
    payload: dict[str, Any]


class NWSClient:
    """NWS client suitable for interactive use, never a settlement authority by itself."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout_s: float = 20, cache_ttl_s: int = 120) -> None:
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.cache_ttl_s = cache_ttl_s
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = Lock()

    def get(self, path: str, params: dict[str, object] | None = None) -> SourceSnapshot:
        url = path if path.startswith("http") else f"{NWS_BASE_URL}{path}"
        key = requests.Request("GET", url, params=params).prepare().url
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
        if cached and now - cached[0] < self.cache_ttl_s:
            with self._lock:
                self._cache.move_to_end(key)
            return self._snapshot(url, cached[1], stale=False)
        error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, headers={"User-Agent": self.user_agent, "Accept": "application/geo+json"}, timeout=self.timeout_s)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("NWS returned a non-object JSON response")
                with self._lock:
                    self._cache[key] = (time.monotonic(), payload)
                    self._cache.move_to_end(key)
                    while len(self._cache) > MAX_CACHE_ENTRIES:
                        self._cache.popitem(last=False)
                return self._snapshot(url, payload, stale=False)
            except (requests.RequestException, ValueError) as exc:
                error = exc
                if attempt < 2:
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"NWS request failed for {url}: {error}") from error

    @staticmethod
    def _snapshot(source: str, payload: dict[str, Any], stale: bool) -> SourceSnapshot:
        props = payload.get("properties") or {}
        timestamp = props.get("timestamp") or props.get("updated")
        source_time = None
        if isinstance(timestamp, str):
            try:
                source_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass
        return SourceSnapshot(source, datetime.now(timezone.utc), source_time, stale, payload)

    def point(self, station: Station) -> SourceSnapshot:
        return self.get(f"/points/{station.latitude},{station.longitude}")

    def hourly_forecast(self, station: Station) -> SourceSnapshot:
        point = self.point(station).payload.get("properties") or {}
        endpoint = point.get("forecastHourly")
        if not isinstance(endpoint, str):
            raise RuntimeError(f"NWS did not provide an hourly endpoint for {station.icao}")
        return self.get(endpoint)

    def grid_forecast(self, station: Station) -> SourceSnapshot:
        return self.get(f"/gridpoints/{station.nws_office}/{station.grid_x},{station.grid_y}")

    def observations(self, station: Station, start: datetime, end: datetime) -> SourceSnapshot:
        return self.get(f"/stations/{station.icao}/observations", {"start": start.astimezone(timezone.utc).isoformat(), "end": end.astimezone(timezone.utc).isoformat(), "limit": 500})


def celsius_to_fahrenheit(value: float | None) -> float | None:
    return None if value is None else value * 9 / 5 + 32


def normalized_observations(snapshot: SourceSnapshot, station: Station) -> list[dict[str, object]]:
    """Deduplicate station observations and preserve source timestamps in local time."""
    rows: dict[datetime, dict[str, object]] = {}
    for feature in snapshot.payload.get("features", []):
        props = feature.get("properties") or {}
        timestamp = props.get("timestamp")
        temp = (props.get("temperature") or {}).get("value")
        if not isinstance(timestamp, str) or temp is None:
            continue
        try:
            observed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            temp_f = celsius_to_fahrenheit(float(temp))
        except (TypeError, ValueError):
            continue
        rows[observed_at] = {"observed_at": observed_at, "local_time": observed_at.astimezone(station.tzinfo), "temperature_f": temp_f, "source": "NWS station observations"}
    return [rows[key] for key in sorted(rows)]


def daily_observation_extremes(rows: list[dict[str, object]], station: Station, target_date: date) -> tuple[float | None, float | None]:
    values = [float(row["temperature_f"]) for row in rows if getattr(row.get("local_time"), "date", lambda: None)() == target_date]
    return (max(values), min(values)) if values else (None, None)


def is_stale(snapshot: SourceSnapshot, max_age: timedelta = timedelta(minutes=90)) -> bool:
    return snapshot.source_time is None or datetime.now(timezone.utc) - snapshot.source_time.astimezone(timezone.utc) > max_age
