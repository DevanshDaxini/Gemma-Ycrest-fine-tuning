import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from prompt_template import render, list_templates


def test_render_default_contains_input():
    result = render("default", input="Weekly sales data here")
    assert "Weekly sales data here" in result


def test_render_default_contains_rules():
    result = render("default", input="data")
    assert "NDJSON" in result
    assert "record_type" in result
    assert "action_id" in result


def test_unknown_template_raises():
    with pytest.raises(FileNotFoundError):
        render("nonexistent_template_xyz", input="data")


def test_list_templates_includes_default():
    templates = list_templates()
    assert "default" in templates


def test_list_templates_returns_list():
    assert isinstance(list_templates(), list)
