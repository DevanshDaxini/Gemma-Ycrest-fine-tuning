"""Tests for /v1/reports endpoints. Stubs mlx_lm and mocks generate_with_retry."""
import sys
import types
import time
from unittest.mock import patch

import pytest

# Stub mlx_lm before app import (same pattern as test_api.py)
_mlx_stub = types.ModuleType("mlx_lm")
_mlx_stub.load = lambda *a, **kw: (object(), object())
_mlx_stub.generate = lambda *a, **kw: '{"record_type": "session_summary"}'
_mlx_stub.stream_generate = lambda *a, **kw: iter(
    [types.SimpleNamespace(text='{"record_type": "session_summary"}')]
)
sys.modules.setdefault("mlx_lm", _mlx_stub)
_sample_utils_stub = types.ModuleType("mlx_lm.sample_utils")
_sample_utils_stub.make_sampler = staticmethod(lambda **kw: (lambda x: x))
sys.modules.setdefault("mlx_lm.sample_utils", _sample_utils_stub)

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
import report_store  # noqa: E402

AUTH = {"Authorization": "Bearer dev-key-12345"}

_VALID_NDJSON = (
    '{"record_type": "session_summary", "week": 1}\n'
    '{"record_type": "highlight", "action_id": "PR_ACHIEVED", "exercise": "Squat"}\n'
    '{"record_type": "recommendation", "action_id": "INCREASE_LOAD", "exercise": "Bench"}'
)

_INVALID_NDJSON = '{"record_type": "highlight", "action_id": "FAKE_ID"}'


@pytest.fixture(autouse=True)
def clear_store():
    report_store._store.clear()
    yield
    report_store._store.clear()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _setup_model(client):
    from model import ModelState
    app.state.model_state = ModelState()
    app.state.model_state.model = object()
    app.state.model_state.tokenizer = _make_stub_tokenizer()
    app.state.model_state.model_loaded = True
    app.state.model_state.adapter_loaded = False


def _make_stub_tokenizer():
    class _Tok:
        def encode(self, text):
            return list(range(len(text.split())))
    return _Tok()


# ── POST /v1/reports ──────────────────────────────────────────────────────────

def test_create_report_valid_output(client):
    _setup_model(client)
    with patch("main.generate_with_retry", return_value=(_VALID_NDJSON, 50, 100, 1, True, [])):
        r = client.post(
            "/v1/reports", headers=AUTH,
            json={"input": "Week 1 data", "template": "default"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["attempts"] == 1
    assert body["id"].startswith("report-")
    assert body["output"] == _VALID_NDJSON
    assert body["errors"] == []
    assert body["usage"]["total_tokens"] == 150


def test_create_report_invalid_output_stored(client):
    _setup_model(client)
    errors = ["Line 1: invalid action_id 'FAKE_ID'"]
    with patch("main.generate_with_retry",
               return_value=(_INVALID_NDJSON, 20, 10, 3, False, errors)):
        r = client.post(
            "/v1/reports", headers=AUTH,
            json={"input": "bad week", "template": "default"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert body["attempts"] == 3
    assert len(body["errors"]) == 1


def test_create_report_unknown_template(client):
    _setup_model(client)
    r = client.post(
        "/v1/reports", headers=AUTH,
        json={"input": "data", "template": "nonexistent_xyz"},
    )
    assert r.status_code == 400


def test_create_report_model_not_loaded(client):
    from model import ModelState
    app.state.model_state = ModelState()  # model_loaded = False
    r = client.post(
        "/v1/reports", headers=AUTH,
        json={"input": "data", "template": "default"},
    )
    assert r.status_code == 503
    _setup_model(client)  # restore


# ── GET /v1/reports ───────────────────────────────────────────────────────────

def test_list_reports_empty(client):
    r = client.get("/v1/reports", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == []


def test_list_reports_returns_created(client):
    _setup_model(client)
    with patch("main.generate_with_retry", return_value=(_VALID_NDJSON, 10, 20, 1, True, [])):
        client.post("/v1/reports", headers=AUTH, json={"input": "data"})
    r = client.get("/v1/reports", headers=AUTH)
    assert r.status_code == 200
    assert len(r.json()) == 1


# ── GET /v1/reports/{id} ──────────────────────────────────────────────────────

def test_get_report_by_id(client):
    _setup_model(client)
    with patch("main.generate_with_retry", return_value=(_VALID_NDJSON, 10, 20, 1, True, [])):
        create_resp = client.post("/v1/reports", headers=AUTH, json={"input": "data"})
    report_id = create_resp.json()["id"]

    r = client.get(f"/v1/reports/{report_id}", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["id"] == report_id


def test_get_nonexistent_report_returns_404(client):
    r = client.get("/v1/reports/report-doesnotexist", headers=AUTH)
    assert r.status_code == 404


# ── DELETE /v1/reports/{id} ───────────────────────────────────────────────────

def test_delete_report(client):
    _setup_model(client)
    with patch("main.generate_with_retry", return_value=(_VALID_NDJSON, 10, 20, 1, True, [])):
        create_resp = client.post("/v1/reports", headers=AUTH, json={"input": "data"})
    report_id = create_resp.json()["id"]

    del_r = client.delete(f"/v1/reports/{report_id}", headers=AUTH)
    assert del_r.status_code == 200
    assert del_r.json()["deleted"] == report_id

    get_r = client.get(f"/v1/reports/{report_id}", headers=AUTH)
    assert get_r.status_code == 404


def test_delete_nonexistent_returns_404(client):
    r = client.delete("/v1/reports/report-ghost", headers=AUTH)
    assert r.status_code == 404


# ── GET /v1/reports/{id}/html ─────────────────────────────────────────────────

def test_report_html_returns_html(client):
    _setup_model(client)
    with patch("main.generate_with_retry", return_value=(_VALID_NDJSON, 10, 20, 1, True, [])):
        create_resp = client.post("/v1/reports", headers=AUTH, json={"input": "data"})
    report_id = create_resp.json()["id"]

    r = client.get(f"/v1/reports/{report_id}/html", headers=AUTH)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<!DOCTYPE html>" in r.text
    assert "Revenue Brief" in r.text
    assert "PR_ACHIEVED" in r.text


def test_report_html_nonexistent_returns_404(client):
    r = client.get("/v1/reports/report-ghost/html", headers=AUTH)
    assert r.status_code == 404


# ── GET /v1/templates ─────────────────────────────────────────────────────────

def test_list_templates(client):
    r = client.get("/v1/templates", headers=AUTH)
    assert r.status_code == 200
    assert "default" in r.json()["templates"]


# ── Rate limit ────────────────────────────────────────────────────────────────

def test_rate_limit_returns_429(client):
    import auth
    # Temporarily set a 1-rpm limiter
    from rate_limiter import RateLimiter
    original = auth._limiter
    auth._limiter = RateLimiter(requests_per_minute=1)
    try:
        client.get("/v1/reports", headers=AUTH)  # consume the 1 allowed request
        r = client.get("/v1/reports", headers=AUTH)
        assert r.status_code == 429
        assert r.json()["error"] == "rate limit exceeded"
    finally:
        auth._limiter = original
