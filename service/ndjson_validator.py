# ndjson_validator.py — validates model output against the revenue brief schema
#
# The fine-tuned model outputs NDJSON: one JSON object per line.
# Three record types exist: session_summary, highlight, recommendation.
# Action IDs are derived from the actual training data — not a hypothetical schema.

import json
from typing import List, Tuple, Dict, Any

# Action IDs present in training data. Both highlight and recommendation records
# use this same closed set — anything outside it is a hallucination.
VALID_ACTION_IDS = {
    "ADD_ACCESSORY_WORK",
    "DECREASE_LOAD",
    "DELOAD_SUGGESTED",
    "FORM_CHECK_NEEDED",
    "INCREASE_LOAD",
    "MAINTAIN_LOAD",
    "MOBILITY_WORK",
    "NUTRITION_CHECK",
    "PLATEAU_DETECTED",
    "PROGRESS_NOTED",
    "PR_ACHIEVED",
    "REST_DAY_RECOMMENDED",
    "SUPERSET_SUGGESTED",
    "TECHNIQUE_REVIEW",
    "VOLUME_DECREASE",
    "VOLUME_INCREASE",
}

# All record_type values the schema recognises.
KNOWN_RECORD_TYPES = {"session_summary", "highlight", "recommendation"}

# record_types that require an action_id field.
RECORD_TYPES_WITH_ACTION_ID = {"highlight", "recommendation"}


def validate_ndjson(content: str) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """Validate a string of NDJSON against the revenue brief schema.

    Checks performed (in order, per line):
      1. Each non-empty line must parse as valid JSON.
      2. Each object must have a "record_type" field.
      3. record_type must be one of the known types.
      4. For highlight and recommendation records: action_id must be present
         and must be one of the valid action IDs.

    Returns:
        (valid, errors, parsed)
        - valid:  True if zero errors were found across all lines.
        - errors: List of human-readable error strings with line numbers.
        - parsed: List of successfully parsed JSON objects (even if some had errors).
    """
    errors: List[str] = []
    parsed: List[Dict[str, Any]] = []

    # Skip blank lines — the model sometimes emits trailing newlines.
    lines = [line for line in content.strip().split("\n") if line.strip()]

    for i, line in enumerate(lines, start=1):
        # Step 1: must be valid JSON.
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: invalid JSON — {e}")
            continue  # can't check further without a parsed object

        parsed.append(obj)

        # Step 2: must have record_type.
        record_type = obj.get("record_type")
        if record_type is None:
            errors.append(f"Line {i}: missing 'record_type' field")
            continue

        # Step 3: record_type must be recognised.
        if record_type not in KNOWN_RECORD_TYPES:
            errors.append(f"Line {i}: unknown record_type '{record_type}'")
            continue

        # Step 4: highlight and recommendation records require a valid action_id.
        # session_summary has no action_id field.
        if record_type in RECORD_TYPES_WITH_ACTION_ID:
            action_id = obj.get("action_id")
            if action_id is None:
                errors.append(f"Line {i}: {record_type} missing 'action_id'")
            elif action_id not in VALID_ACTION_IDS:
                errors.append(f"Line {i}: invalid action_id '{action_id}'")

    return len(errors) == 0, errors, parsed
