"""Production HTTP API for the PolyWeather dashboard.

Serves the same JSON payload the local Vite dev server produces by shelling
out to scripts/dashboard_payload.py, but as a real WSGI app suitable for a
hosted deployment (see DEPLOY.md). Run locally with:

    .venv/Scripts/python.exe -m flask --app server run --port 8000
"""

from __future__ import annotations

import os
from datetime import date

from flask import Flask, jsonify, request
from flask_cors import CORS

from polyweather.dashboard_payload import dashboard_today, payload

app = Flask(__name__)

# Only the configured dashboard origin(s) may call this API in production.
# Set ALLOWED_ORIGINS to a comma-separated list (e.g. the deployed dashboard
# URL); unset it locally to allow any origin during development.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS")
CORS(app, origins=_allowed_origins.split(",") if _allowed_origins else "*")


@app.get("/api/dashboard")
def dashboard() -> tuple:
    raw_date = request.args.get("date")
    try:
        target_date = date.fromisoformat(raw_date) if raw_date else dashboard_today()
    except ValueError:
        return jsonify({"error": f"Invalid date {raw_date!r}. Use YYYY-MM-DD."}), 400
    try:
        return jsonify(payload(target_date))
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
