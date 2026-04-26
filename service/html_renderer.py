"""Render a list of parsed NDJSON records as a clean HTML report for leadership."""
import html as _html
from datetime import datetime, timezone
from typing import Any, Dict, List


# Action IDs that indicate positive outcomes — rendered with green badge.
_POSITIVE_ACTION_IDS = {
    "PR_ACHIEVED", "PROGRESS_NOTED", "INCREASE_LOAD", "VOLUME_INCREASE",
    "SUPERSET_SUGGESTED", "ADD_ACCESSORY_WORK",
}

# Action IDs that indicate caution / review needed — rendered with amber badge.
_CAUTION_ACTION_IDS = {
    "FORM_CHECK_NEEDED", "TECHNIQUE_REVIEW", "PLATEAU_DETECTED",
    "NUTRITION_CHECK", "MOBILITY_WORK",
}

# Everything else renders with the default blue badge.


def _badge_color(action_id: str) -> str:
    if action_id in _POSITIVE_ACTION_IDS:
        return "#16a34a"   # green
    if action_id in _CAUTION_ACTION_IDS:
        return "#d97706"   # amber
    return "#2563eb"       # blue (decrease/deload/rest)


def _ts_to_human(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _e(text: Any) -> str:
    """HTML-escape a value."""
    return _html.escape(str(text))


def _render_summary(obj: Dict[str, Any]) -> str:
    skip = {"record_type"}
    rows = "".join(
        f"<tr><td style='padding:6px 12px;color:#6b7280;font-size:13px'>{_e(k)}</td>"
        f"<td style='padding:6px 12px;font-size:13px'>{_e(v)}</td></tr>"
        for k, v in obj.items() if k not in skip
    )
    return (
        "<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;"
        "padding:16px 20px;margin-bottom:20px'>"
        "<h2 style='margin:0 0 12px;font-size:16px;color:#0369a1'>Session Summary</h2>"
        f"<table style='border-collapse:collapse'>{rows}</table></div>"
    )


def _render_action_record(obj: Dict[str, Any], record_type: str) -> str:
    action_id = obj.get("action_id", "")
    color = _badge_color(action_id)
    title = "Highlight" if record_type == "highlight" else "Recommendation"
    skip = {"record_type", "action_id"}
    detail_rows = "".join(
        f"<div style='margin-top:6px;font-size:13px;color:#374151'>"
        f"<span style='color:#9ca3af;margin-right:6px'>{_e(k)}:</span>{_e(v)}</div>"
        for k, v in obj.items() if k not in skip
    )
    return (
        "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:8px;"
        "padding:14px 18px;margin-bottom:12px'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
        f"<span style='font-size:11px;font-weight:600;color:#6b7280;text-transform:uppercase'>"
        f"{_e(title)}</span>"
        f"<span style='background:{color};color:#fff;font-size:11px;font-weight:600;"
        f"padding:2px 8px;border-radius:4px'>{_e(action_id)}</span></div>"
        f"{detail_rows}</div>"
    )


def _render_unknown(obj: Dict[str, Any]) -> str:
    import json
    return (
        "<div style='background:#fef9c3;border:1px solid #fde68a;border-radius:8px;"
        "padding:12px 16px;margin-bottom:12px;font-family:monospace;font-size:12px'>"
        f"{_e(json.dumps(obj))}</div>"
    )


def render_report_html(
    report_id: str,
    created_at: int,
    template_name: str,
    valid: bool,
    attempts: int,
    records: List[Dict[str, Any]],
) -> str:
    summaries = [r for r in records if r.get("record_type") == "session_summary"]
    highlights = [r for r in records if r.get("record_type") == "highlight"]
    recommendations = [r for r in records if r.get("record_type") == "recommendation"]
    others = [r for r in records if r.get("record_type") not in
              {"session_summary", "highlight", "recommendation"}]

    body_parts: List[str] = []

    for r in summaries:
        body_parts.append(_render_summary(r))

    if highlights:
        body_parts.append(
            "<h3 style='margin:24px 0 12px;font-size:15px;color:#111827'>Highlights</h3>"
        )
        body_parts.extend(_render_action_record(r, "highlight") for r in highlights)

    if recommendations:
        body_parts.append(
            "<h3 style='margin:24px 0 12px;font-size:15px;color:#111827'>Recommendations</h3>"
        )
        body_parts.extend(_render_action_record(r, "recommendation") for r in recommendations)

    for r in others:
        body_parts.append(_render_unknown(r))

    if not records:
        body_parts.append(
            "<p style='color:#9ca3af;font-style:italic'>No records in this report.</p>"
        )

    validity_badge = (
        "<span style='background:#16a34a;color:#fff;font-size:11px;font-weight:600;"
        "padding:2px 8px;border-radius:4px'>VALID</span>"
        if valid else
        "<span style='background:#dc2626;color:#fff;font-size:11px;font-weight:600;"
        "padding:2px 8px;border-radius:4px'>INVALID</span>"
    )

    meta = (
        f"<div style='font-size:12px;color:#9ca3af;margin-bottom:24px'>"
        f"<span style='margin-right:16px'>ID: {_e(report_id)}</span>"
        f"<span style='margin-right:16px'>Generated: {_e(_ts_to_human(created_at))}</span>"
        f"<span style='margin-right:16px'>Template: {_e(template_name)}</span>"
        f"<span style='margin-right:16px'>Attempts: {_e(str(attempts))}</span>"
        f"{validity_badge}</div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Revenue Brief — {_e(report_id)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f9fafb; margin: 0; padding: 32px; color: #111827; }}
    .container {{ max-width: 860px; margin: 0 auto; }}
    h1 {{ font-size: 22px; font-weight: 700; margin: 0 0 4px; color: #111827; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Revenue Brief</h1>
    {meta}
    {"".join(body_parts)}
  </div>
</body>
</html>"""
