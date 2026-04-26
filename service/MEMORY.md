# Conversation Memory

## What it does

When a `session_id` is included in a request, the service stores the user message
and model response after each turn. On the next request with the same `session_id`,
all prior turns are prepended to the prompt so the model sees the full conversation.

This means the model can reference earlier workout logs in later requests.

---

## How to use it

Add `"session_id"` to any request. Use any string — typically a client ID or UUID.

**Turn 1 — first workout log:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "jane-2024",
    "messages": [{"role": "user", "content": "Workout Log — Jane — 2024-04-25\nDuration: 45 min\n\n• Squat: 8x60kg | 10x60kg | 6x60kg\n• Leg Curl: 8x40kg | 10x40kg\n\nNotes: felt strong"}],
    "max_tokens": 512
  }'
```

**Turn 2 — second workout log (model sees turn 1 automatically):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "jane-2024",
    "messages": [{"role": "user", "content": "Workout Log — Jane — 2024-05-02\nDuration: 50 min\n\n• Squat: 10x62.5kg | 8x62.5kg | 6x62.5kg\n• Leg Curl: 10x42.5kg | 8x42.5kg\n\nNotes: pushed harder than last week"}],
    "max_tokens": 512
  }'
```

**Clear a session:**
```bash
curl -X DELETE http://localhost:8000/v1/sessions/jane-2024 \
  -H "Authorization: Bearer dev-key-12345"
```

---

## Storage backends

The default backend is an in-memory Python dict (`session_store.py`).
It requires zero setup and works on Mac, but all sessions are lost when the
server restarts.

### Option A — In-memory (default, already active)

No setup needed. Data lost on restart.

```
# No changes required — this is already running.
```

---

### Option B — SQLite (persistent, single-process)

Survives server restarts. Good for a single-server deployment without Redis.

**Install:**
```bash
pip install aiosqlite
```

**Replace `session_store.py` with:**
```python
import sqlite3
import json
from pathlib import Path

DB_PATH = Path("sessions.db")
MAX_TURNS = 10

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts INTEGER NOT NULL
        )
    """)
    conn.commit()
    return conn

def get_history(session_id: str) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM sessions WHERE session_id=? ORDER BY ts",
            (session_id,)
        ).fetchall()
    return [{"role": r, "content": c} for r, c in rows]

def append_turn(session_id: str, user_content: str, assistant_content: str) -> None:
    ts = __import__("time").time_ns()
    with _conn() as conn:
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?)", (session_id, "user", user_content, ts))
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?)", (session_id, "assistant", assistant_content, ts + 1))
        conn.commit()
        # Trim to MAX_TURNS pairs
        count = conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id=?", (session_id,)).fetchone()[0]
        if count > MAX_TURNS * 2:
            cutoff = conn.execute(
                "SELECT ts FROM sessions WHERE session_id=? ORDER BY ts LIMIT 1 OFFSET ?",
                (session_id, count - MAX_TURNS * 2)
            ).fetchone()[0]
            conn.execute("DELETE FROM sessions WHERE session_id=? AND ts < ?", (session_id, cutoff))
            conn.commit()

def clear_session(session_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        conn.commit()

def session_exists(session_id: str) -> bool:
    with _conn() as conn:
        return conn.execute(
            "SELECT 1 FROM sessions WHERE session_id=? LIMIT 1", (session_id,)
        ).fetchone() is not None
```

---

### Option C — Redis (persistent, multi-process, production)

Required if you run multiple uvicorn workers or multiple server instances.

**Install:**
```bash
pip install redis
```

**Set env var:**
```
REDIS_URL=redis://localhost:6379
```

**Replace `session_store.py` with:**
```python
import json
import os
import redis

MAX_TURNS = 10
_r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

def _key(session_id: str) -> str:
    return f"session:{session_id}"

def get_history(session_id: str) -> list:
    raw = _r.lrange(_key(session_id), 0, -1)
    return [json.loads(x) for x in raw]

def append_turn(session_id: str, user_content: str, assistant_content: str) -> None:
    key = _key(session_id)
    pipe = _r.pipeline()
    pipe.rpush(key, json.dumps({"role": "user", "content": user_content}))
    pipe.rpush(key, json.dumps({"role": "assistant", "content": assistant_content}))
    pipe.ltrim(key, -(MAX_TURNS * 2), -1)
    pipe.execute()

def clear_session(session_id: str) -> None:
    _r.delete(_key(session_id))

def session_exists(session_id: str) -> bool:
    return _r.exists(_key(session_id)) > 0
```

**Start Redis (Mac with Homebrew):**
```bash
brew install redis
brew services start redis
```

**Start Redis (Linux server):**
```bash
sudo apt install redis-server
sudo systemctl start redis
```

---

## Context window limit

The model has an 8,192 token context window. Each workout log turn is ~300–600 tokens.
`MAX_TURNS = 10` in `session_store.py` limits stored pairs to ~10 exchanges, which
stays safely within the window. Lower it if you see truncation errors.

---

## Important caveat

This model was fine-tuned on **single-turn** workout log → NDJSON pairs. It was not
trained on multi-turn conversations. Passing it session history may improve consistency
(it can reference prior volume numbers) but it may also produce unexpected output.

Test with real data before relying on multi-turn behavior in production.
