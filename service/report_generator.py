import logging
from typing import Tuple

from model import ModelState, run_inference, format_messages
from ndjson_validator import validate_ndjson

logger = logging.getLogger(__name__)


def generate_with_retry(
    state: ModelState,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 1500,
    max_retries: int = 3,
) -> Tuple[str, int, int, int, bool, list[str]]:
    """Run inference with automatic retry when output fails NDJSON validation.

    On each failed attempt, the bad output and a correction request are appended
    to the message history so the model sees its own mistake and the specific errors.
    This is more effective than re-running the same prompt, as the model can target
    the exact lines that failed.

    Args:
        state: Loaded model state.
        messages: Conversation messages — typically a single user message built from
                  a prompt template.
        temperature: Sampling temperature (0 = greedy, recommended for structured output).
        max_tokens: Max tokens to generate per attempt.
        max_retries: Total attempts before giving up (1 = no retry).

    Returns:
        (output, prompt_tokens, completion_tokens, attempts, valid, errors)
        - output: Last generated text (valid or not).
        - prompt_tokens / completion_tokens: Summed across all attempts.
        - attempts: How many attempts were made.
        - valid: True if the final output passed validation.
        - errors: Validation errors from the last attempt (empty if valid).
    """
    current_messages = list(messages)
    total_pt = 0
    total_ct = 0
    last_output = ""
    last_errors: list[str] = []

    for attempt in range(1, max_retries + 1):
        prompt = format_messages(current_messages)
        output, pt, ct = run_inference(
            state, prompt, temperature, max_tokens, raw_prompt=True
        )
        total_pt += pt
        total_ct += ct
        last_output = output

        valid, errors, _ = validate_ndjson(output)
        if valid:
            logger.info("Report valid on attempt %d/%d", attempt, max_retries)
            return output, total_pt, total_ct, attempt, True, []

        last_errors = errors
        logger.warning(
            "Attempt %d/%d produced invalid NDJSON (%d errors)%s",
            attempt, max_retries, len(errors),
            " — retrying" if attempt < max_retries else " — giving up",
        )

        if attempt < max_retries:
            # Cap error list so the correction prompt doesn't balloon.
            error_summary = "\n".join(errors[:5])
            current_messages = current_messages + [
                {"role": "assistant", "content": output},
                {"role": "user", "content": (
                    f"That output had validation errors:\n{error_summary}\n\n"
                    "Fix the output. Return valid NDJSON only — one JSON object per line, "
                    "no other text."
                )},
            ]

    return last_output, total_pt, total_ct, max_retries, False, last_errors
