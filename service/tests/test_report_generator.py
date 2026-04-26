"""Tests for report_generator. Uses unittest.mock to avoid needing a real model."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

from model import ModelState
from report_generator import generate_with_retry

_VALID_OUTPUT = '{"record_type": "session_summary"}\n{"record_type": "highlight", "action_id": "PR_ACHIEVED"}'
_INVALID_OUTPUT = '{"record_type": "highlight", "action_id": "MADE_UP_ID"}'


def _make_state():
    s = ModelState()
    s.model_loaded = True
    return s


def test_valid_output_returns_first_attempt():
    with patch("report_generator.run_inference", return_value=(_VALID_OUTPUT, 10, 20)):
        output, pt, ct, attempts, valid, errors = generate_with_retry(
            _make_state(),
            messages=[{"role": "user", "content": "give me a report"}],
        )
    assert valid is True
    assert attempts == 1
    assert errors == []
    assert output == _VALID_OUTPUT


def test_invalid_output_retries():
    call_outputs = [
        (_INVALID_OUTPUT, 10, 20),
        (_VALID_OUTPUT, 10, 20),
    ]
    with patch("report_generator.run_inference", side_effect=call_outputs):
        _, _, _, attempts, valid, _ = generate_with_retry(
            _make_state(),
            messages=[{"role": "user", "content": "give me a report"}],
            max_retries=3,
        )
    assert valid is True
    assert attempts == 2


def test_max_retries_exceeded_returns_invalid():
    with patch("report_generator.run_inference", return_value=(_INVALID_OUTPUT, 10, 20)):
        _, _, _, attempts, valid, errors = generate_with_retry(
            _make_state(),
            messages=[{"role": "user", "content": "give me a report"}],
            max_retries=2,
        )
    assert valid is False
    assert attempts == 2
    assert len(errors) > 0


def test_tokens_accumulate_across_attempts():
    with patch("report_generator.run_inference", return_value=(_INVALID_OUTPUT, 10, 20)):
        _, pt, ct, _, _, _ = generate_with_retry(
            _make_state(),
            messages=[{"role": "user", "content": "test"}],
            max_retries=3,
        )
    assert pt == 30   # 10 × 3 attempts
    assert ct == 60   # 20 × 3 attempts


def test_single_attempt_no_retry():
    with patch("report_generator.run_inference", return_value=(_INVALID_OUTPUT, 5, 5)):
        _, _, _, attempts, valid, _ = generate_with_retry(
            _make_state(),
            messages=[{"role": "user", "content": "test"}],
            max_retries=1,
        )
    assert attempts == 1
    assert valid is False
