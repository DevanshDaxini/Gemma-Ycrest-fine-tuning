#!/usr/bin/env python3
"""
Demo script — runs both fine-tuned models and prints results.
Usage: python demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent / "phase2_ndjson"))

from mlx_utils import format_gemma_prompt, greedy_generate, load_model_with_adapter
from schema import parse_ndjson, validate_record

MODEL = "mlx-community/gemma-3-1b-it-4bit"
DIVIDER = "=" * 60


# ── PHASE 1 EXAMPLES ────────────────────────────────────────────
# These words come from the model's training vocabulary.
# The model learned a fixed mapping: each nonsense word maps to
# a specific output word. It also learned to handle multiple
# words in one prompt.

PHASE1_EXAMPLES = [
    {
        "label": "Single word",
        "prompt": "Map: flanoix",
        "expected": "pumol",
    },
    {
        "label": "Single word",
        "prompt": "Map: flooero",
        "expected": "tijof",
    },
    {
        "label": "Two words (composition)",
        "prompt": "Map: flanoix flooero",
        "expected": "pumol tijof",
    },
    {
        "label": "Three words (stretch)",
        "prompt": "Map: flanoix flooero craa",
        "expected": "pumol tijof mibol",
    },
]


# ── PHASE 2 EXAMPLES ────────────────────────────────────────────
# These are workout logs in different writing styles.
# The model learned to parse them and emit structured NDJSON.

PHASE2_EXAMPLES = [
    {
        "label": "Terse format",
        "prompt": "Alex 45min BP:3x8@80kg SQ:4x5@100kg",
    },
    {
        "label": "Bullet format",
        "prompt": (
            "Workout Log — Jordan — 2024-03-15\n"
            "Duration: 60 min\n\n"
            "• Bench Press: 5x100kg | 5x100kg | 5x97.5kg\n"
            "• Deadlift: 3x140kg | 3x140kg\n"
            "• Bicep Curl: 10x30kg | 10x30kg | 10x30kg\n\n"
            "Notes: felt easy on everything, could do more"
        ),
    },
    {
        "label": "Prose format",
        "prompt": (
            "Sam trained on 2024-05-20 for 18 minutes. "
            "Did Squat for 4 reps at 80kg then 4 reps at 80kg. "
            "Exhausted, rough day."
        ),
    },
]


def run_phase1(model, tok):
    print(f"\n{DIVIDER}")
    print("  PHASE 1 — Nonsense Word Mapping")
    print(f"{DIVIDER}")
    print("  The model learned a secret lookup table of 200 made-up")
    print("  words. Give it any word from that table and it returns")
    print("  the mapped output. Works for 1, 2, or 3 words at once.")
    print()

    for ex in PHASE1_EXAMPLES:
        prompt_text = format_gemma_prompt(ex["prompt"])
        output, stats = greedy_generate(model, tok, prompt_text, max_tokens=64)
        correct = output.strip().lower() == ex["expected"].strip().lower()
        status = "CORRECT" if correct else "WRONG"

        print(f"  [{ex['label']}]")
        print(f"  Input    : {ex['prompt']}")
        print(f"  Expected : {ex['expected']}")
        print(f"  Got      : {output.strip()}")
        print(f"  Result   : {status}  |  {stats['tokens_per_sec']:.0f} tok/sec  |  {stats['elapsed_s']:.1f}s")
        print()


def run_phase2(model, tok):
    print(f"\n{DIVIDER}")
    print("  PHASE 2 — Workout Log → Structured NDJSON")
    print(f"{DIVIDER}")
    print("  The model learned to read workout logs written in any")
    print("  style and convert them into structured JSON records —")
    print("  one record per line, ready to parse or convert to HTML.")
    print()

    for ex in PHASE2_EXAMPLES:
        prompt_text = format_gemma_prompt(ex["prompt"])
        output, stats = greedy_generate(model, tok, prompt_text, max_tokens=256)

        lines = parse_ndjson(output)
        valid = sum(1 for ok, _ in lines if ok)
        schema_ok = sum(
            1 for ok, obj in lines
            if ok and isinstance(obj, dict) and validate_record(obj)[0]
        )

        print(f"  [{ex['label']}]")
        print(f"  Input:\n    {ex['prompt'][:120].replace(chr(10), chr(10) + '    ')}")
        print()
        print("  Output (NDJSON):")
        for ok, obj in lines:
            if ok:
                import json
                print(f"    {json.dumps(obj)}")
            else:
                print(f"    [INVALID JSON] {obj}")
        print()
        print(f"  Lines: {len(lines)}  |  JSON valid: {valid}  |  Schema valid: {schema_ok}"
              f"  |  {stats['tokens_per_sec']:.0f} tok/sec  |  {stats['elapsed_s']:.1f}s")
        print()


def main():
    print(f"\n{DIVIDER}")
    print("  Gemma 3 1B — Fine-Tuning Demo")
    print("  Two tasks, one model, trained on Apple Silicon (M4)")
    print(f"{DIVIDER}")

    # Phase 1
    print("\nLoading Phase 1 model...")
    p1_adapter = str(Path(__file__).parent / "phase1_toy" / "adapters")
    model, tok = load_model_with_adapter(MODEL, p1_adapter)
    run_phase1(model, tok)

    # Phase 2 — reload with different adapter
    print("\nLoading Phase 2 model...")
    p2_adapter = str(Path(__file__).parent / "phase2_ndjson" / "adapters")
    model, tok = load_model_with_adapter(MODEL, p2_adapter)
    run_phase2(model, tok)

    print(f"{DIVIDER}")
    print("  Demo complete.")
    print(f"{DIVIDER}\n")


if __name__ == "__main__":
    main()
