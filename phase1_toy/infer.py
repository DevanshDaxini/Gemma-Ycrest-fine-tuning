#!/usr/bin/env python3
"""
Phase 1 inference: nonsense word mapping.
Usage:
  python infer.py --prompt "Map: florbix guavella" --mode greedy
  python infer.py --prompt "Map: florbix" --mode distribution --top-k 5
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "shared"))
from mlx_utils import (
    distribution_generate,
    format_gemma_prompt,
    greedy_generate,
    load_model_with_adapter,
)

DEFAULT_MODEL   = "mlx-community/gemma-3-1b-it-4bit"
DEFAULT_ADAPTER = str(HERE / "adapters")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 inference")
    p.add_argument("--prompt",  required=True, help="e.g. 'Map: word1 word2'")
    p.add_argument("--mode",    default="greedy", choices=["greedy", "distribution"])
    p.add_argument("--top-k",   type=int, default=5)
    p.add_argument("--n-tokens",type=int, default=10,
                   help="tokens to show in distribution mode")
    p.add_argument("--model",   default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    p.add_argument("--no-adapter", action="store_true",
                   help="run base model without adapter")
    args = p.parse_args()

    adapter = None if args.no_adapter else args.adapter
    model, tokenizer = load_model_with_adapter(args.model, adapter)

    prompt_text = format_gemma_prompt(args.prompt)

    if args.mode == "greedy":
        output, stats = greedy_generate(model, tokenizer, prompt_text, max_tokens=64)
        print(f"\nInput : {args.prompt}")
        print(f"Output: {output.strip()}")
        print(f"\n  Input tokens : {stats['input_tokens']}")
        print(f"  Output tokens: {stats['output_tokens']}")
        print(f"  Time         : {stats['elapsed_s']:.1f}s")
        print(f"  Speed        : {stats['tokens_per_sec']:.1f} tok/sec")

    else:  # distribution
        print(f"\nInput : {args.prompt}")
        print(f"Top-{args.top_k} token distribution for first {args.n_tokens} steps:\n")
        steps = distribution_generate(
            model, tokenizer, prompt_text,
            top_k=args.top_k, n_tokens=args.n_tokens,
        )
        for i, step in enumerate(steps, 1):
            print(f"Step {i:2d}:")
            for rank, (tok, prob, lp) in enumerate(step, 1):
                bar = "█" * int(prob * 30)
                print(f"  {rank}. {tok!r:18s}  prob={prob:.4f}  logp={lp:.4f}  {bar}")
        print()
        greedy_output = "".join(s[0][0] for s in steps)
        print(f"Greedy continuation (first {args.n_tokens} tokens): {greedy_output!r}")


if __name__ == "__main__":
    main()
