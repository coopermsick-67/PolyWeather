"""Authoritative, market-oriented settlement-station registry.

The configured location is deliberately the *settlement station*, rather than
the city centroid. Adding a location is data-only: provide a validated
``Station`` record (or load JSON records with :func:`load_station_registry`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_MARKET_TYPES = ("daily_high", "daily_low", "temperature_bracket", "threshold_gte", "threshold_lte")


@dataclass(frozen=True)
class Station:
    """A named weather market and its exact, configured settlement station."""

    slug: str
    icao: str
    ghcn_id: str
    name: str
    display_name: str
    display_note: str
    latitude: float
    longitude: float
    timezone: str
    nws_office: str
    grid_x: int
    grid_y: int
    aliases: tuple[str, ...] = ()
    market_types: tuple[str, ...] = DEFAULT_MARKET_TYPES
    source_priority: tuple[str, ...] = ("nws_climate_report", "ncei_daily", "nws_observation", "metar")
    data_quality_status: str = "VERIFIED"
    last_successful_observation_time: str | None = None

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def to_public_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["stationId"] = self.icao
        value["forecastGrid"] = {"office": self.nws_office, "x": self.grid_x, "y": self.grid_y}
        value.pop("icao")
        value.pop("ghcn_id")
        return value


def _station(
    slug: str, icao: str, ghcn_id: str, name: str, display_name: str, note: str,
    latitude: float, longitude: float, timezone: str, office: str, grid_x: int, grid_y: int,
    *aliases: str,
) -> Station:
    return Station(slug, icao, ghcn_id, name, display_name, note, latitude, longitude, timezone, office, grid_x, grid_y, aliases)


# Grid metadata was resolved from api.weather.gov/points on 2026-09-01. The
# live NWS client still follows the points endpoint at runtime, so an NWS grid
# migration does not silently make this registry stale.
_REGISTRY: tuple[Station, ...] = (
    _station("new-york-city", "KNYC", "USW00094728", "Central Park, NY", "New York City", "Central Park", 40.7829, -73.9654, "America/New_York", "OKX", 34, 45, "new york", "nyc", "central park"),
    _station("chicago", "KMDW", "USW00014819", "Chicago Midway Airport, IL", "Chicago", "Midway; not O'Hare", 41.7868, -87.7522, "America/Chicago", "LOT", 72, 69, "chicago midway", "midway", "ord", "o'hare"),
    _station("miami", "KMIA", "USW00012839", "Miami International Airport, FL", "Miami", "Miami International", 25.7959, -80.2870, "America/New_York", "MFL", 106, 51, "mia", "miami international"),
    _station("austin", "KAUS", "USW00013958", "Austin-Bergstrom International Airport, TX", "Austin", "Austin-Bergstrom", 30.1945, -97.6699, "America/Chicago", "EWX", 159, 88, "aus", "austin bergstrom"),
    _station("los-angeles", "KLAX", "USW00023174", "Los Angeles International Airport, CA", "Los Angeles", "Los Angeles International", 33.9416, -118.4085, "America/Los_Angeles", "LOX", 148, 41, "la", "lax", "los angeles international"),
    _station("denver", "KDEN", "USW00003017", "Denver International Airport, CO", "Denver", "Denver International", 39.8561, -104.6737, "America/Denver", "BOU", 74, 66, "den", "denver international"),
    _station("phoenix", "KPHX", "USW00023183", "Phoenix Sky Harbor International Airport, AZ", "Phoenix", "Phoenix Sky Harbor", 33.4342, -112.0116, "America/Phoenix", "PSR", 161, 57, "phx", "sky harbor"),
    _station("philadelphia", "KPHL", "USW00013739", "Philadelphia International Airport, PA", "Philadelphia", "Philadelphia International", 39.8729, -75.2437, "America/New_York", "PHI", 48, 75, "philly", "phl"),
    _station("houston", "KHOU", "USW00012960", "William P. Hobby Airport, TX", "Houston", "Hobby; not Bush/IAH", 29.6454, -95.2789, "America/Chicago", "HGX", 66, 89, "hobby", "iah", "bush intercontinental"),
    _station("minneapolis", "KMSP", "USW00014922", "Minneapolis-Saint Paul International Airport, MN", "Minneapolis", "Minneapolis-Saint Paul", 44.8820, -93.2218, "America/Chicago", "MPX", 110, 68, "minneapolis saint paul", "msp"),
    _station("oklahoma-city", "KOKC", "USW00013967", "Will Rogers World Airport, OK", "Oklahoma City", "Will Rogers", 35.3931, -97.6007, "America/Chicago", "OUN", 94, 90, "okc", "will rogers"),
    _station("san-francisco", "KSFO", "USW00023234", "San Francisco International Airport, CA", "San Francisco", "San Francisco International", 37.6188, -122.3750, "America/Los_Angeles", "MTR", 85, 98, "sf", "sfo", "san francisco international"),
    _station("washington-dc", "KDCA", "USW00013743", "Ronald Reagan Washington National Airport, DC", "Washington, D.C.", "Reagan National", 38.8512, -77.0402, "America/New_York", "LWX", 97, 69, "washington dc", "dc", "dca", "reagan"),
    _station("boston", "KBOS", "USW00014739", "Boston Logan International Airport, MA", "Boston", "Logan International", 42.3656, -71.0096, "America/New_York", "BOX", 73, 102, "bos", "logan"),
    _station("dallas", "KDFW", "USW00003927", "Dallas/Fort Worth International Airport, TX", "Dallas", "DFW; not Love Field", 32.8998, -97.0403, "America/Chicago", "FWD", 80, 109, "dfw", "love field", "dallas fort worth"),
    _station("seattle", "KSEA", "USW00024233", "Seattle-Tacoma International Airport, WA", "Seattle", "Seattle-Tacoma", 47.4502, -122.3088, "America/Los_Angeles", "SEW", 124, 61, "sea", "seatac", "seattle tacoma"),
    _station("las-vegas", "KLAS", "USW00023169", "Harry Reid International Airport, NV", "Las Vegas", "Harry Reid", 36.0801, -115.1522, "America/Los_Angeles", "VEF", 122, 94, "vegas", "las", "harry reid"),
    _station("atlanta", "KATL", "USW00013874", "Hartsfield-Jackson Atlanta International Airport, GA", "Atlanta", "Hartsfield-Jackson", 33.6407, -84.4277, "America/New_York", "FFC", 50, 82, "atl", "hartsfield"),
    _station("san-antonio", "KSAT", "USW00012921", "San Antonio International Airport, TX", "San Antonio", "San Antonio International", 29.5337, -98.4698, "America/Chicago", "EWX", 127, 59, "sat", "san antonio international"),
    _station("new-orleans", "KMSY", "USW00012916", "Louis Armstrong New Orleans International Airport, LA", "New Orleans", "Louis Armstrong", 29.9934, -90.2580, "America/Chicago", "LIX", 60, 90, "nola", "msy", "louis armstrong"),
)

STATIONS: dict[str, Station] = {station.icao: station for station in _REGISTRY}
STATIONS_BY_SLUG: dict[str, Station] = {station.slug: station for station in _REGISTRY}


def require_station(value: str) -> Station:
    """Resolve ICAO, market slug, display name, or configured alias safely."""
    normalized = value.strip().casefold()
    for station in _REGISTRY:
        candidates = (station.icao, station.slug, station.display_name, station.name, *station.aliases)
        if normalized in {candidate.casefold() for candidate in candidates}:
            return station
    choices = ", ".join(STATIONS)
    raise ValueError(f"Unknown settlement station {value!r}. Choose one of: {choices}.")


def station_metadata() -> list[dict[str, object]]:
    return [station.to_public_dict() for station in _REGISTRY]


def load_station_registry(path: str | Path) -> dict[str, Station]:
    """Load additional configuration records without changing UI code.

    The loader rejects duplicate IDs/slugs and malformed timezones instead of
    guessing a settlement station.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Station registry configuration must be a JSON list.")
    loaded: dict[str, Station] = {}
    for record in records:
        try:
            station = Station(
                slug=str(record["slug"]), icao=str(record["icao"]).upper(), ghcn_id=str(record["ghcn_id"]),
                name=str(record["name"]), display_name=str(record["display_name"]), display_note=str(record["display_note"]),
                latitude=float(record["latitude"]), longitude=float(record["longitude"]), timezone=str(record["timezone"]),
                nws_office=str(record["nws_office"]), grid_x=int(record["grid_x"]), grid_y=int(record["grid_y"]),
                aliases=tuple(record.get("aliases", ())), market_types=tuple(record.get("market_types", DEFAULT_MARKET_TYPES)),
            )
            _ = station.tzinfo
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid station registry entry: {record!r}") from exc
        if station.icao in STATIONS or station.icao in loaded or station.slug in STATIONS_BY_SLUG or any(s.slug == station.slug for s in loaded.values()):
            raise ValueError(f"Duplicate configured settlement station: {station.icao}/{station.slug}.")
        loaded[station.icao] = station
    return loaded
