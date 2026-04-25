#!/usr/bin/env python3
"""
Phase 1 evaluation.
Reports accuracy for 4 categories:
  singleton             – memorisation baseline
  seen_composition      – 2-word pairs seen during training
  held_out_composition  – 2-word pairs NOT seen during training (key metric)
  triple                – 3-word stretch generalisation
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "shared"))
from mlx_utils import format_gemma_prompt, greedy_generate, load_model_with_adapter

DEFAULT_MODEL   = "mlx-community/gemma-3-1b-it-4bit"
DEFAULT_ADAPTER = str(HERE / "adapters")


def _exact(pred: str, expected: str) -> bool:
    pred = pred.replace("<end_of_turn>", "").strip()
    return pred.lower() == expected.strip().lower()


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 eval")
    p.add_argument("--model",    default=DEFAULT_MODEL)
    p.add_argument("--adapter",  default=DEFAULT_ADAPTER)
    p.add_argument("--data",     default=str(HERE / "data" / "test.jsonl"))
    p.add_argument("--limit",    type=int, default=None, help="cap examples per category")
    p.add_argument("--no-adapter", action="store_true")
    args = p.parse_args()

    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    adapter = None if args.no_adapter else args.adapter
    model, tok = load_model_with_adapter(args.model, adapter)

    buckets: dict[str, list] = defaultdict(list)
    for ex in examples:
        cat = ex.get("category", "singleton")
        inp = ex.get("input")
        exp = ex.get("expected")
        if inp is None or exp is None:
            continue
        prompt = format_gemma_prompt(inp)
        pred, _stats = greedy_generate(model, tok, prompt, max_tokens=64)
        correct = _exact(pred, exp)
        buckets[cat].append({"correct": correct, "pred": pred.strip(), "expected": exp})
        status = "✓" if correct else "✗"
        print(f"[{status}] [{cat}] expected={exp!r}  got={pred.strip()!r}")

    print("\n" + "=" * 56)
    print("Phase 1 Evaluation Results")
    print("=" * 56)
    ORDER = ["singleton", "seen_composition", "held_out_composition", "triple"]
    for cat in ORDER:
        rows = buckets.get(cat, [])
        if not rows:
            continue
        n = len(rows)
        n_ok = sum(r["correct"] for r in rows)
        pct = 100 * n_ok / n
        bar = "█" * int(pct / 5)
        print(f"  {cat:30s}: {n_ok:3d}/{n:3d} = {pct:5.1f}%  {bar}")

        failures = [r for r in rows if not r["correct"]][:2]
        for f in failures:
            print(f"      ✗ pred={f['pred']!r}  exp={f['expected']!r}")

    all_rows = [r for rows in buckets.values() for r in rows]
    if all_rows:
        overall = 100 * sum(r["correct"] for r in all_rows) / len(all_rows)
        print(f"\n  {'OVERALL':30s}: {overall:.1f}%  ({len(all_rows)} examples)")


if __name__ == "__main__":
    main()
