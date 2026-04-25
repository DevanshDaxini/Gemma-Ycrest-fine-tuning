#!/usr/bin/env python3
"""
Phase 2 inference: workout log → NDJSON report.
Usage:
  python infer.py --prompt "Alex trained 45min..." --mode greedy
  python infer.py --prompt "..." --mode distribution --top-k 5
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE))
from mlx_utils import (
    distribution_generate,
    format_gemma_prompt,
    greedy_generate,
    load_model_with_adapter,
)
from schema import ACTION_IDS, parse_ndjson, validate_record

DEFAULT_MODEL   = "mlx-community/gemma-3-1b-it-4bit"
DEFAULT_ADAPTER = str(HERE / "adapters")


def pretty_ndjson(raw: str) -> None:
    """Parse, validate, and pretty-print NDJSON output."""
    lines = parse_ndjson(raw)
    if not lines:
        print("  (no output lines)")
        return

    valid_json = invalid_json = valid_schema = invalid_schema = 0
    for ok, obj in lines:
        if not ok:
            print(f"  [INVALID JSON] {obj}")
            invalid_json += 1
            continue
        valid_json += 1

        schema_ok, err = validate_record(obj)
        rt = obj.get("record_type", "?")
        aid = obj.get("action_id", "")

        if not schema_ok:
            print(f"  [SCHEMA ERR] {err}")
            print(f"    raw: {json.dumps(obj)}")
            invalid_schema += 1
        else:
            valid_schema += 1
            flag = " ⚠ UNKNOWN action_id" if aid and aid not in ACTION_IDS else ""
            print(f"  [{rt.upper()}]{flag}")
            for k, v in obj.items():
                if k != "record_type":
                    print(f"    {k}: {v}")

    total = valid_json + invalid_json
    print(f"\n  Lines: {total}  |  JSON valid: {valid_json}  "
          f"|  Schema valid: {valid_schema}  |  Errors: {invalid_json + invalid_schema}")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 inference")
    p.add_argument("--prompt",   required=True)
    p.add_argument("--mode",     default="greedy", choices=["greedy", "distribution"])
    p.add_argument("--top-k",    type=int, default=5)
    p.add_argument("--n-tokens", type=int, default=10)
    p.add_argument("--model",    default=DEFAULT_MODEL)
    p.add_argument("--adapter",  default=DEFAULT_ADAPTER)
    p.add_argument("--no-adapter", action="store_true")
    args = p.parse_args()

    adapter = None if args.no_adapter else args.adapter
    model, tokenizer = load_model_with_adapter(args.model, adapter)

    prompt_text = format_gemma_prompt(args.prompt)

    if args.mode == "greedy":
        raw, stats = greedy_generate(model, tokenizer, prompt_text, max_tokens=256)
        print("\n=== NDJSON Output ===")
        pretty_ndjson(raw)
        print(f"\n  Input tokens : {stats['input_tokens']}")
        print(f"  Output tokens: {stats['output_tokens']}")
        print(f"  Time         : {stats['elapsed_s']:.1f}s")
        print(f"  Speed        : {stats['tokens_per_sec']:.1f} tok/sec")

    else:
        print(f"\nTop-{args.top_k} distribution for first {args.n_tokens} steps:\n")
        steps = distribution_generate(
            model, tokenizer, prompt_text,
            top_k=args.top_k, n_tokens=args.n_tokens,
        )
        for i, step in enumerate(steps, 1):
            print(f"Step {i:2d}:")
            for rank, (tok, prob, lp) in enumerate(step, 1):
                bar = "█" * int(prob * 25)
                print(f"  {rank}. {tok!r:20s}  prob={prob:.4f}  logp={lp:.4f}  {bar}")
        print()


if __name__ == "__main__":
    main()
