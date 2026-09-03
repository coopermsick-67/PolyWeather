"""Production HTTP API for the PolyWeather dashboard.

Serves the same JSON payload the local Vite dev server produces by shelling
out to scripts/dashboard_payload.py, but as a real WSGI app suitable for a
hosted deployment (see DEPLOY.md). Run locally with:

    .venv/Scripts/python.exe -m flask --app server run --port 8000
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date

from flask import Flask, jsonify, request
from flask_cors import CORS

from polyweather.dashboard_payload import dashboard_today, payload

# Without a configured handler, error-level logs from polyweather.* modules
# (e.g. the per-station fault isolation in dashboard_payload.py) hit
# Python's root logger, which has no handler by default under gunicorn and
# so silently drops them instead of reaching the platform's log stream.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = Flask(__name__)

# Only the configured dashboard origin(s) may call this API in production.
# Set ALLOWED_ORIGINS to a comma-separated list (e.g. the deployed dashboard
# URL); unset it locally to allow any origin during development.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS")
CORS(app, origins=_allowed_origins.split(",") if _allowed_origins else "*")


@app.after_request
def _apply_response_headers(response):
    # This endpoint serves live forecast data; an intermediary or browser
    # cache serving a stale response would silently show an outdated
    # forecast, which matters more here than for a typical read endpoint.
    response.headers["Cache-Control"] = "no-store"
    # Baseline hardening headers; this API has no HTML/script surface of its
    # own, but these cost nothing and help if it is ever proxied or embedded.
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# payload() does live upstream fetches plus model inference; several
# clients (tabs, polling) requesting the same date within a few seconds
# would otherwise each pay that full cost and each add load on NWS. This
# cache is per-process (fine under gunicorn: each worker independently
# bounds its own duplicate-request cost) and intentionally short so it
# never masks the live per-refresh behavior the frontend depends on.
_PAYLOAD_CACHE_TTL_S = 20.0
_payload_cache: dict[date, tuple[float, dict]] = {}
_payload_cache_lock = threading.Lock()


def _clear_payload_cache() -> None:
    """Test-only hook to reset in-process cache state between test cases."""
    with _payload_cache_lock:
        _payload_cache.clear()


def _cached_payload(target_date: date) -> dict:
    now = time.monotonic()
    with _payload_cache_lock:
        cached = _payload_cache.get(target_date)
        if cached and now - cached[0] < _PAYLOAD_CACHE_TTL_S:
            return cached[1]
    result = payload(target_date)
    with _payload_cache_lock:
        _payload_cache[target_date] = (now, result)
        # Bound memory: only today/near-future dates are ever requested in
        # practice (validate_dashboard_date rejects the rest), but this
        # keeps a long-running worker's cache from growing unbounded.
        if len(_payload_cache) > 16:
            oldest_key = min(_payload_cache, key=lambda key: _payload_cache[key][0])
            _payload_cache.pop(oldest_key, None)
    return result


@app.get("/api/dashboard")
def dashboard() -> tuple:
    raw_date = request.args.get("date")
    try:
        target_date = date.fromisoformat(raw_date) if raw_date else dashboard_today()
    except ValueError:
        return jsonify({"error": f"Invalid date {raw_date!r}. Use YYYY-MM-DD."}), 400
    force_refresh = request.args.get("refresh") == "1"
    try:
        if force_refresh:
            with _payload_cache_lock:
                _payload_cache.pop(target_date, None)
            return jsonify(payload(target_date))
        return jsonify(_cached_payload(target_date))
    except ValueError as exc:
        # Caller-facing validation errors (bad date range, etc.) are safe to surface.
        return jsonify({"error": str(exc)}), 400
    except Exception:  # noqa: BLE001 - API boundary: never leak internals in a 500
        app.logger.exception("Failed to build dashboard payload for %s", target_date)
        return jsonify({"error": "The forecast service is temporarily unavailable. Try again shortly."}), 500


@app.get("/healthz")
def healthz() -> tuple:
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1", port=8000, debug=debug)
