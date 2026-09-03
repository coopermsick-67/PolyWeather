"""Integration tests for the Flask API surface in server.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    server._clear_payload_cache()
    with server.app.test_client() as test_client:
        yield test_client
    server._clear_payload_cache()


def test_healthz_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_dashboard_rejects_invalid_date(client):
    response = client.get("/api/dashboard?date=not-a-date")
    assert response.status_code == 400
    body = response.get_json()
    assert "error" in body


def test_dashboard_surfaces_validation_error_as_400(client):
    with patch("server.payload", side_effect=ValueError("Choose a date within range.")):
        response = client.get("/api/dashboard?date=2020-01-01")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Choose a date within range."


def test_dashboard_hides_internal_error_details_on_500(client):
    with patch("server.payload", side_effect=RuntimeError("secret internal path /etc/whatever")):
        response = client.get("/api/dashboard?date=2020-01-01")
    assert response.status_code == 500
    body = response.get_json()
    assert "secret internal path" not in body["error"]


def test_dashboard_success_returns_payload(client):
    with patch("server.payload", return_value={"forecasts": []}):
        response = client.get("/api/dashboard")
    assert response.status_code == 200
    assert response.get_json() == {"forecasts": []}


def test_dashboard_response_has_no_store_and_security_headers(client):
    with patch("server.payload", return_value={"forecasts": []}):
        response = client.get("/api/dashboard")
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_dashboard_caches_repeated_requests_for_the_same_date(client):
    with patch("server.payload", return_value={"forecasts": []}) as mocked:
        client.get("/api/dashboard?date=2020-01-01")
        client.get("/api/dashboard?date=2020-01-01")
    assert mocked.call_count == 1


def test_dashboard_refresh_param_bypasses_the_cache(client):
    with patch("server.payload", return_value={"forecasts": []}) as mocked:
        client.get("/api/dashboard?date=2020-01-01")
        client.get("/api/dashboard?date=2020-01-01&refresh=1")
    assert mocked.call_count == 2


def test_dashboard_does_not_cache_a_failed_request(client):
    with patch("server.payload", side_effect=RuntimeError("boom")) as mocked:
        client.get("/api/dashboard?date=2020-01-01")
    with patch("server.payload", return_value={"forecasts": []}) as mocked2:
        response = client.get("/api/dashboard?date=2020-01-01")
    assert response.status_code == 200
    assert mocked.call_count == 1
    assert mocked2.call_count == 1
