import pytest
from ndjson_validator import validate_ndjson


def test_valid_session_summary():
    ok, errors, parsed = validate_ndjson('{"record_type": "session_summary"}')
    assert ok
    assert parsed[0]["record_type"] == "session_summary"
    assert errors == []


def test_valid_recommendation():
    ok, errors, _ = validate_ndjson('{"record_type": "recommendation", "action_id": "INCREASE_LOAD"}')
    assert ok
    assert errors == []


def test_valid_highlight():
    ok, errors, _ = validate_ndjson('{"record_type": "highlight", "action_id": "PR_ACHIEVED", "exercise": "Squat", "detail": "New PR: 100kg x 5 reps"}')
    assert ok
    assert errors == []


def test_all_valid_action_ids():
    action_ids = [
        "ADD_ACCESSORY_WORK", "DECREASE_LOAD", "DELOAD_SUGGESTED",
        "FORM_CHECK_NEEDED", "INCREASE_LOAD", "MAINTAIN_LOAD",
        "MOBILITY_WORK", "NUTRITION_CHECK", "PLATEAU_DETECTED",
        "PROGRESS_NOTED", "PR_ACHIEVED", "REST_DAY_RECOMMENDED",
        "SUPERSET_SUGGESTED", "TECHNIQUE_REVIEW", "VOLUME_DECREASE",
        "VOLUME_INCREASE",
    ]
    for action_id in action_ids:
        ok, errors, _ = validate_ndjson(
            f'{{"record_type": "recommendation", "action_id": "{action_id}"}}'
        )
        assert ok, f"{action_id} should be valid, got: {errors}"


def test_invalid_action_id():
    ok, errors, _ = validate_ndjson('{"record_type": "recommendation", "action_id": "FAKE_ACTION"}')
    assert not ok
    assert "Line 1" in errors[0]
    assert "FAKE_ACTION" in errors[0]


def test_invalid_action_id_on_highlight():
    ok, errors, _ = validate_ndjson('{"record_type": "highlight", "action_id": "NOT_REAL"}')
    assert not ok
    assert "NOT_REAL" in errors[0]


def test_missing_action_id_recommendation():
    ok, errors, _ = validate_ndjson('{"record_type": "recommendation"}')
    assert not ok
    assert "action_id" in errors[0]


def test_missing_action_id_highlight():
    ok, errors, _ = validate_ndjson('{"record_type": "highlight"}')
    assert not ok
    assert "action_id" in errors[0]


def test_missing_record_type():
    ok, errors, _ = validate_ndjson('{"foo": "bar"}')
    assert not ok
    assert "record_type" in errors[0]


def test_unknown_record_type():
    ok, errors, _ = validate_ndjson('{"record_type": "unknown_type"}')
    assert not ok
    assert "unknown_type" in errors[0]


def test_invalid_json():
    ok, errors, _ = validate_ndjson("not json at all")
    assert not ok
    assert "invalid JSON" in errors[0]


def test_multiline_all_valid():
    content = (
        '{"record_type": "session_summary", "date": "2024-01-01", "user": "Jane"}\n'
        '{"record_type": "highlight", "action_id": "PR_ACHIEVED", "exercise": "Squat"}\n'
        '{"record_type": "recommendation", "action_id": "MAINTAIN_LOAD"}'
    )
    ok, errors, parsed = validate_ndjson(content)
    assert ok
    assert len(parsed) == 3


def test_multiline_partial_error_line_number():
    content = (
        '{"record_type": "session_summary"}\n'
        '{"record_type": "recommendation", "action_id": "BAD"}'
    )
    ok, errors, parsed = validate_ndjson(content)
    assert not ok
    assert len(errors) == 1
    assert "Line 2" in errors[0]
    assert len(parsed) == 2


def test_empty_lines_ignored():
    content = '{"record_type": "session_summary"}\n\n\n'
    ok, errors, parsed = validate_ndjson(content)
    assert ok
    assert len(parsed) == 1


def test_empty_string():
    ok, errors, parsed = validate_ndjson("")
    assert ok
    assert parsed == []
    assert errors == []
