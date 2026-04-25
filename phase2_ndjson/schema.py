"""
Phase 2 NDJSON schema.
Record types: session_summary | highlight | recommendation
18 closed-set action IDs.
"""
from typing import Literal

from pydantic import BaseModel, field_validator

ACTION_IDS: frozenset[str] = frozenset([
    "PROGRESS_NOTED",
    "VOLUME_INCREASE",
    "INCREASE_LOAD",
    "DECREASE_LOAD",
    "FORM_CHECK_NEEDED",
    "DELOAD_SUGGESTED",
    "PR_ACHIEVED",
    "PLATEAU_DETECTED",
    "REST_DAY_RECOMMENDED",
    "VOLUME_DECREASE",
    "MAINTAIN_LOAD",
    "ADD_ACCESSORY_WORK",
    "TECHNIQUE_REVIEW",
    "INCREASE_REPS",
    "DECREASE_REPS",
    "SUPERSET_SUGGESTED",
    "MOBILITY_WORK",
    "NUTRITION_CHECK",
])


class SessionSummary(BaseModel):
    record_type: Literal["session_summary"]
    date: str
    user: str
    total_volume_kg: float
    exercise_count: int
    session_rating: int  # 1-5

    @field_validator("session_rating")
    @classmethod
    def rating_range(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError(f"session_rating must be 1-5, got {v}")
        return v


class Highlight(BaseModel):
    record_type: Literal["highlight"]
    action_id: str
    exercise: str
    detail: str

    @field_validator("action_id")
    @classmethod
    def valid_action(cls, v: str) -> str:
        if v not in ACTION_IDS:
            raise ValueError(f"Unknown action_id: {v!r}")
        return v


class Recommendation(BaseModel):
    record_type: Literal["recommendation"]
    action_id: str
    exercise: str
    reason: str
    priority: Literal["low", "medium", "high"]

    @field_validator("action_id")
    @classmethod
    def valid_action(cls, v: str) -> str:
        if v not in ACTION_IDS:
            raise ValueError(f"Unknown action_id: {v!r}")
        return v


_SCHEMA_MAP = {
    "session_summary": SessionSummary,
    "highlight": Highlight,
    "recommendation": Recommendation,
}

RECORD_TYPES = frozenset(_SCHEMA_MAP.keys())


def validate_record(record: dict) -> tuple[bool, str]:
    """Validate one parsed NDJSON record. Returns (ok, error_msg)."""
    rt = record.get("record_type")
    if rt not in _SCHEMA_MAP:
        return False, f"Unknown record_type: {rt!r}"
    try:
        _SCHEMA_MAP[rt](**record)
        return True, ""
    except Exception as e:
        return False, str(e)


def parse_ndjson(text: str) -> list[tuple[bool, dict | str]]:
    """
    Parse multi-line NDJSON string.
    Returns list of (parsed_ok, obj_or_error) per non-empty line.
    """
    results = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            import json
            obj = json.loads(line)
            results.append((True, obj))
        except Exception as e:
            results.append((False, str(e)))
    return results
