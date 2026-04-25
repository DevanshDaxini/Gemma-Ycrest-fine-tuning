#!/usr/bin/env python3
"""
Phase 2 tiered evaluation.
Tier 1 – NDJSON validity   : every output line parses as JSON
Tier 2 – Action ID validity : all action_id fields in 18-item closed set
Tier 3 – Schema adherence   : each record matches its record_type schema
Tier 4 – Semantic accuracy  : spot-check on ≤10 examples vs deterministic rules
Also prints: confusion-matrix of predicted action_ids (invented vs valid).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE))
from mlx_utils import format_gemma_prompt, greedy_generate, load_model_with_adapter
from schema import ACTION_IDS, parse_ndjson, validate_record

DEFAULT_MODEL   = "mlx-community/gemma-3-1b-it-4bit"
DEFAULT_ADAPTER = str(HERE / "adapters")


def _run_inference(model, tok, test_data: list[dict]) -> list[dict]:
    results = []
    for ex in test_data:
        inp = ex.get("input", "")
        prompt = format_gemma_prompt(inp)
        raw, _stats = greedy_generate(model, tok, prompt, max_tokens=256)
        results.append({"raw": raw, "example": ex})
    return results


def _tier_report(results: list[dict]) -> tuple[dict, Counter]:
    t1 = t2 = t3 = 0
    action_counter: Counter = Counter()
    semantic_cases = []

    for res in results:
        raw = res["raw"]
        lines = parse_ndjson(raw)
        if not lines:
            continue

        # Tier 1
        all_json_ok = all(ok for ok, _ in lines)
        if all_json_ok:
            t1 += 1

        objs = [obj for ok, obj in lines if ok]

        # Tier 2
        aids = [obj.get("action_id") for obj in objs if "action_id" in obj]
        t2_ok = all(a in ACTION_IDS for a in aids) if aids else True
        if all_json_ok and t2_ok:
            t2 += 1
        for a in aids:
            if a:
                action_counter[a] += 1

        # Tier 3
        schema_ok = all(validate_record(obj)[0] for obj in objs)
        if all_json_ok and t2_ok and schema_ok:
            t3 += 1

        # Tier 4 spot-check (semantic)
        if len(semantic_cases) < 10:
            semantic_cases.append(res)

    return {"t1": t1, "t2": t2, "t3": t3,
            "n": len(results), "semantic": semantic_cases}, action_counter


def _semantic_spot_check(cases: list[dict]) -> float:
    """
    Heuristic semantic check: compare predicted action_ids to expected_records.
    Returns fraction of cases where >=50% of expected action_ids are present.
    """
    ok = 0
    for case in cases:
        ex = case["example"]
        expected = ex.get("expected_records", [])
        raw = case["raw"]
        lines = parse_ndjson(raw)
        pred_aids = {obj.get("action_id") for _, obj in lines
                     if isinstance(obj, dict) and "action_id" in obj}
        exp_aids  = {r.get("action_id") for r in expected if "action_id" in r}
        if not exp_aids:
            ok += 1
            continue
        overlap = pred_aids & exp_aids
        if len(overlap) / len(exp_aids) >= 0.5:
            ok += 1
    return ok / len(cases) if cases else 0.0


def _confusion_table(counter: Counter) -> None:
    print("\n  Action ID usage (predicted):")
    print(f"  {'Action ID':<30}  {'Count':>5}  Status")
    print(f"  {'-'*30}  {'-'*5}  ------")
    for aid, cnt in sorted(counter.items(), key=lambda x: -x[1]):
        status = "✓ valid" if aid in ACTION_IDS else "✗ INVENTED"
        print(f"  {aid:<30}  {cnt:5d}  {status}")
    invented = [a for a in counter if a not in ACTION_IDS]
    if not invented:
        print("  No invented action IDs detected.")
    else:
        print(f"\n  ⚠ Invented IDs: {invented}")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 eval")
    p.add_argument("--model",    default=DEFAULT_MODEL)
    p.add_argument("--adapter",  default=DEFAULT_ADAPTER)
    p.add_argument("--data",     default=str(HERE / "data" / "test.jsonl"))
    p.add_argument("--limit",    type=int, default=None)
    p.add_argument("--no-adapter", action="store_true")
    args = p.parse_args()

    test_data = []
    with open(args.data) as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    if args.limit:
        test_data = test_data[:args.limit]

    adapter = None if args.no_adapter else args.adapter
    model, tok = load_model_with_adapter(args.model, adapter)

    print(f"\nRunning inference on {len(test_data)} test examples...")
    results = _run_inference(model, tok, test_data)

    metrics, action_counter = _tier_report(results)
    n = metrics["n"]
    if n == 0:
        print("No results.")
        return

    print("\n" + "=" * 56)
    print("Phase 2 Tiered Evaluation")
    print("=" * 56)
    labels = [
        ("Tier 1 - NDJSON valid",    metrics["t1"]),
        ("Tier 2 - Action ID valid", metrics["t2"]),
        ("Tier 3 - Schema adherent", metrics["t3"]),
    ]
    for label, count in labels:
        pct = 100 * count / n
        bar = "█" * int(pct / 4)
        print(f"  {label:<28}: {count:3d}/{n} = {pct:5.1f}%  {bar}")

    sem = _semantic_spot_check(metrics["semantic"])
    print(f"  {'Tier 4 - Semantic (<=10)':<28}: {sem*100:.1f}%  (spot-check)")

    _confusion_table(action_counter)


if __name__ == "__main__":
    main()
