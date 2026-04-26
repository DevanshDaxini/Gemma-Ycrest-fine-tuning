import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import report_store as rs
from report_store import StoredReport, new_report_id


def _make_report(**overrides) -> StoredReport:
    defaults = dict(
        id=new_report_id(),
        created_at=int(time.time()),
        input_text="weekly data",
        template_name="default",
        output='{"record_type": "session_summary"}',
        valid=True,
        errors=[],
        attempts=1,
        prompt_tokens=10,
        completion_tokens=20,
    )
    defaults.update(overrides)
    return StoredReport(**defaults)


def setup_function():
    # clear store between tests
    rs._store.clear()


def test_save_and_get():
    r = _make_report(id="report-abc")
    rs.save_report(r)
    fetched = rs.get_report("report-abc")
    assert fetched is not None
    assert fetched.id == "report-abc"


def test_get_nonexistent_returns_none():
    assert rs.get_report("report-does-not-exist") is None


def test_list_reports_sorted_newest_first():
    r1 = _make_report(id="r1", created_at=1000)
    r2 = _make_report(id="r2", created_at=2000)
    r3 = _make_report(id="r3", created_at=3000)
    for r in [r1, r3, r2]:
        rs.save_report(r)
    listed = rs.list_reports()
    assert [r.id for r in listed] == ["r3", "r2", "r1"]


def test_delete_report():
    r = _make_report(id="report-del")
    rs.save_report(r)
    assert rs.delete_report("report-del") is True
    assert rs.get_report("report-del") is None


def test_delete_nonexistent_returns_false():
    assert rs.delete_report("report-ghost") is False


def test_new_report_id_unique():
    ids = {new_report_id() for _ in range(100)}
    assert len(ids) == 100
