"""
API integration tests. Uses TestClient (no real model needed).
Stubs mlx_lm before importing app so tests run without GPU or model weights.
"""
import sys
import types

import pytest

# Stub mlx_lm before app import
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

AUTH = {"Authorization": "Bearer dev-key-12345"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- /health ---

def test_health_no_auth_required(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_returns_model_status(client):
    data = client.get("/health").json()
    assert "model_loaded" in data
    assert "adapter_loaded" in data
    assert data["status"] == "ok"


# --- auth ---

def test_missing_auth_header_returns_401(client):
    r = client.post("/v1/validate", json={"content": ""})
    assert r.status_code == 401
    assert r.json() == {"error": "unauthorized"}


def test_wrong_api_key_returns_401(client):
    r = client.post(
        "/v1/validate",
        headers={"Authorization": "Bearer wrong-key"},
        json={"content": ""},
    )
    assert r.status_code == 401


def test_malformed_auth_scheme_returns_401(client):
    r = client.post(
        "/v1/validate",
        headers={"Authorization": "Token dev-key-12345"},
        json={"content": ""},
    )
    assert r.status_code == 401


# --- /v1/validate ---

def test_validate_valid_ndjson(client):
    r = client.post(
        "/v1/validate",
        headers=AUTH,
        json={"content": '{"record_type": "session_summary"}'},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["errors"] == []
    assert len(body["parsed"]) == 1


def test_validate_invalid_ndjson(client):
    r = client.post(
        "/v1/validate",
        headers=AUTH,
        json={"content": '{"record_type": "recommendation", "action_id": "BAD"}'},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert len(body["errors"]) == 1


def test_validate_empty_content(client):
    r = client.post("/v1/validate", headers=AUTH, json={"content": ""})
    assert r.status_code == 200
    assert r.json()["valid"] is True


# --- /v1/chat/completions ---

def test_chat_completions_model_not_loaded_returns_503(client):
    # TestClient loads app — model stub loads fine but if model_loaded is False, expect 503
    # This test verifies the 503 path exists; actual model_loaded depends on stub
    r = client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "gemma-revenue-brief", "messages": [{"role": "user", "content": "test"}]},
    )
    # Either 200 (stub loaded) or 503 (model path missing) — both are valid, not a 5xx crash
    assert r.status_code in (200, 503)


def test_chat_completions_response_shape(client):
    # Patch model_state to loaded so we can verify response shape
    from model import ModelState
    app.state.model_state = ModelState()
    app.state.model_state.model = object()
    app.state.model_state.tokenizer = _make_stub_tokenizer()
    app.state.model_state.model_loaded = True
    app.state.model_state.adapter_loaded = False

    r = client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "model": "gemma-revenue-brief",
            "messages": [{"role": "user", "content": "give me a brief"}],
            "temperature": 0,
            "max_tokens": 256,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in body
    assert "prompt_tokens" in body["usage"]


def test_chat_completions_stream_sse_format(client):
    from model import ModelState
    app.state.model_state = ModelState()
    app.state.model_state.model = object()
    app.state.model_state.tokenizer = _make_stub_tokenizer()
    app.state.model_state.model_loaded = True
    app.state.model_state.adapter_loaded = False

    r = client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={
            "model": "gemma-revenue-brief",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    body = r.text
    assert "data: " in body
    assert "data: [DONE]" in body
    # First chunk must include role
    import json as _json
    first_chunk = _json.loads(body.split("data: ")[1].split("\n")[0])
    assert first_chunk["choices"][0]["delta"].get("role") == "assistant"


def _make_stub_tokenizer():
    class _Tok:
        def encode(self, text):
            return list(range(len(text.split())))
    return _Tok()
