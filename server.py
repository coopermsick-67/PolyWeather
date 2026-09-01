"""Production HTTP API for the PolyWeather dashboard.

Serves the same JSON payload the local Vite dev server produces by shelling
out to scripts/dashboard_payload.py, but as a real WSGI app suitable for a
hosted deployment (see DEPLOY.md). Run locally with:

    .venv/Scripts/python.exe -m flask --app server run --port 8000
"""

from __future__ import annotations

from datetime import date

from flask import Flask, jsonify, request
from flask_cors import CORS

from polyweather.dashboard_payload import dashboard_today, payload

app = Flask(__name__)
CORS(app)


@app.get("/api/dashboard")
def dashboard() -> tuple:
    raw_date = request.args.get("date")
    try:
        target_date = date.fromisoformat(raw_date) if raw_date else dashboard_today()
    except ValueError:
        return jsonify({"error": f"Invalid date {raw_date!r}. Use YYYY-MM-DD."}), 400
    try:
        return jsonify(payload(target_date))
    except Exception as exc:  # noqa: BLE001 - API boundary: never leak a bare 500
        app.logger.exception("Failed to build dashboard payload for %s", target_date)
        return jsonify({"error": str(exc)}), 500


@app.get("/healthz")
def healthz() -> tuple:
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
