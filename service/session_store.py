# session_store.py — in-memory conversation history store
#
# Keeps prior turns per session_id so the model can see previous workout logs
# and produce consistent follow-up responses.
#
# This implementation uses a plain Python dict — zero dependencies, works on Mac.
# Data is lost when the server restarts. See MEMORY.md for persistent backends.

from collections import defaultdict
from typing import Dict, List

# Maximum number of user+assistant pairs to retain per session.
# Each pair consumes ~300-600 tokens. Gemma 3 1B has an 8k token context window,
# so 10 pairs leaves room for the new request without overflowing.
MAX_TURNS = 10

# session_id → list of {"role": ..., "content": ...} dicts (alternating user/assistant)
_store: Dict[str, List[dict]] = defaultdict(list)


def get_history(session_id: str) -> List[dict]:
    """Return all stored messages for this session, oldest first."""
    return list(_store[session_id])


def append_turn(session_id: str, user_content: str, assistant_content: str) -> None:
    """Store a completed user+assistant exchange, trimming oldest pairs if needed."""
    history = _store[session_id]
    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": assistant_content})

    # Each turn = 2 messages. Drop oldest pair when over limit.
    max_messages = MAX_TURNS * 2
    if len(history) > max_messages:
        _store[session_id] = history[-max_messages:]


def clear_session(session_id: str) -> None:
    """Delete all stored history for this session."""
    _store.pop(session_id, None)


def session_exists(session_id: str) -> bool:
    return session_id in _store and len(_store[session_id]) > 0
