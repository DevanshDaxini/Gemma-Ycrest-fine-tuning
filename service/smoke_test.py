#!/usr/bin/env python3
"""
End-to-end smoke test. Starts uvicorn, waits for model load, tests all endpoints.

Usage:
    cd service && python smoke_test.py [--timeout 300]

Requires the model weights to be accessible (downloads from HuggingFace if needed).
"""
import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:8765"
HOST = "127.0.0.1"
PORT = 8765
API_KEY = os.environ.get("API_KEY", "dev-key-12345")
AUTH = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def _request(method: str, path: str, body=None, headers=None, stream=False):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    payload = json.dumps(body).encode() if body else None
    conn.request(method, path, body=payload, headers=h)
    resp = conn.getresponse()
    if stream:
        raw = resp.read().decode()
        return resp.status, resp.getheaders(), raw
    data = json.loads(resp.read().decode())
    return resp.status, resp.getheaders(), data


def _check(label: str, condition: bool, detail: str = ""):
    icon = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon} {label}{suffix}")
    return condition


def wait_for_model(timeout: int) -> bool:
    deadline = time.time() + timeout
    print(f"Waiting up to {timeout}s for model to load (may download weights)...")
    while time.time() < deadline:
        try:
            status, _, data = _request("GET", "/health")
            if status == 200 and data.get("model_loaded"):
                print(f"  Model ready. adapter_loaded={data.get('adapter_loaded')}")
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run_tests() -> int:
    failures = 0

    # --- /health ---
    print("\n[GET /health]")
    status, _, data = _request("GET", "/health")
    failures += not _check("status 200", status == 200, str(status))
    failures += not _check("model_loaded true", data.get("model_loaded") is True)
    failures += not _check("status field ok", data.get("status") == "ok")

    # --- auth ---
    print("\n[Auth]")
    status, _, _ = _request("POST", "/v1/validate", body={"content": ""})
    failures += not _check("no auth → 401", status == 401, str(status))

    status, _, _ = _request("POST", "/v1/validate", body={"content": ""},
                             headers={"Authorization": "Bearer wrong", "Content-Type": "application/json"})
    failures += not _check("wrong key → 401", status == 401, str(status))

    # --- /v1/validate ---
    print("\n[POST /v1/validate]")
    status, _, data = _request("POST", "/v1/validate",
                                body={"content": '{"record_type": "session_summary"}'},
                                headers=AUTH)
    failures += not _check("status 200", status == 200, str(status))
    failures += not _check("valid true", data.get("valid") is True)

    status, _, data = _request("POST", "/v1/validate",
                                body={"content": '{"record_type": "recommendation", "action_id": "BAD"}'},
                                headers=AUTH)
    failures += not _check("invalid → valid false", data.get("valid") is False)

    # --- /v1/chat/completions (non-streaming) ---
    print("\n[POST /v1/chat/completions]")
    payload = {
        "model": "gemma-revenue-brief",
        "messages": [{"role": "user", "content": "Generate a brief for: Client: Jane. Goal: fat loss."}],
        "temperature": 0,
        "max_tokens": 256,
    }
    status, _, data = _request("POST", "/v1/chat/completions", body=payload, headers=AUTH)
    failures += not _check("status 200", status == 200, str(status))
    if status == 200:
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        failures += not _check("non-empty content", len(content) > 0, repr(content[:80]))
        failures += not _check("has usage", "usage" in data)
        failures += not _check("id prefix", data.get("id", "").startswith("chatcmpl-"))
        print(f"    output preview: {content[:120]!r}")

    # --- /v1/chat/completions (streaming) ---
    print("\n[POST /v1/chat/completions stream=true]")
    payload["stream"] = True
    status, headers_raw, raw = _request("POST", "/v1/chat/completions", body=payload,
                                          headers=AUTH, stream=True)
    headers_dict = {k.lower(): v for k, v in headers_raw}
    failures += not _check("status 200", status == 200, str(status))
    failures += not _check("content-type SSE", "text/event-stream" in headers_dict.get("content-type", ""))
    failures += not _check("contains data: prefix", "data: " in raw)
    failures += not _check("ends with [DONE]", "data: [DONE]" in raw)
    chunks = [l[6:] for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
    assembled = "".join(json.loads(c)["choices"][0]["delta"].get("content", "") for c in chunks)
    failures += not _check("assembled content non-empty", len(assembled) > 0, repr(assembled[:80]))

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=300, help="Seconds to wait for model load")
    parser.add_argument("--no-server", action="store_true", help="Skip starting server (already running on port 8765)")
    args = parser.parse_args()

    service_dir = Path(__file__).parent

    proc = None
    if not args.no_server:
        env = {**os.environ, "PORT": str(PORT)}
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", HOST, "--port", str(PORT)],
            cwd=service_dir,
            env=env,
        )
        print(f"Started uvicorn pid={proc.pid}")

    try:
        if not wait_for_model(args.timeout):
            print(f"\n{FAIL} Model did not load within {args.timeout}s")
            sys.exit(1)

        failures = run_tests()
        print(f"\n{'All tests passed' if failures == 0 else f'{failures} test(s) FAILED'}")
        sys.exit(0 if failures == 0 else 1)
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
            print("Server stopped.")


if __name__ == "__main__":
    main()
