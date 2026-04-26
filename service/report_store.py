import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StoredReport:
    id: str
    created_at: int          # Unix timestamp
    input_text: str
    template_name: str
    output: str              # raw NDJSON string
    valid: bool
    errors: List[str]
    attempts: int
    prompt_tokens: int
    completion_tokens: int


_store: Dict[str, StoredReport] = {}


def new_report_id() -> str:
    return f"report-{uuid.uuid4().hex}"


def save_report(report: StoredReport) -> None:
    _store[report.id] = report


def get_report(report_id: str) -> Optional[StoredReport]:
    return _store.get(report_id)


def list_reports() -> List[StoredReport]:
    """Return all reports, newest first."""
    return sorted(_store.values(), key=lambda r: r.created_at, reverse=True)


def delete_report(report_id: str) -> bool:
    """Delete a report. Returns True if it existed."""
    if report_id in _store:
        del _store[report_id]
        return True
    return False
