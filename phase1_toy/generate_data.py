#!/usr/bin/env python3
"""
Phase 1 data generator: nonsense word mapping task.
  - 200 singleton pairs (fruit-like word → nonsense word)
  - 150 two-word composition pairs (training)
  - 50 held-out two-word composition pairs (test)
  - 25 triple-word stretch pairs (test)
  - 10% of training pool → valid.jsonl; rest → train.jsonl
Seeded for reproducibility (SEED=42).
"""
import itertools
import json
import random
from pathlib import Path

SEED = 42
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

# Syllable pools for fruit-like input words
_INIT = ["bl","cr","dr","fl","gl","gr","pl","pr","sl","sn",
         "sp","st","str","sw","tr","br","fr","sk","sc","cl"]
_VOWEL = ["a","e","i","o","u","ai","ei","au","oo","ee"]
_MID   = ["w","r","l","n","m","b","d","g","v","z","nd","ng"]
_END   = ["berry","fruit","ine","ula","ix","el","on","a","o",
          "etta","ella","ium","ax","ero","ile","oid","orn"]

_CONS  = list("bcdfghjklmnprstvwxz")
_SVOW  = list("aeiou")


def _make_fruit(rng: random.Random) -> str:
    w = rng.choice(_INIT) + rng.choice(_VOWEL)
    if rng.random() > 0.45:
        w += rng.choice(_MID) + rng.choice(_VOWEL)
    w += rng.choice(_END)
    return w


def _make_output(rng: random.Random) -> str:
    c1, v1, c2 = rng.choice(_CONS), rng.choice(_SVOW), rng.choice(_CONS)
    if rng.random() > 0.4:
        v2, c3 = rng.choice(_SVOW), rng.choice(_CONS)
        return c1 + v1 + c2 + v2 + c3
    return c1 + v1 + c2


def _gen_unique(fn, n: int, rng: random.Random,
                min_len: int = 4, max_len: int = 13) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    tries = 0
    while len(out) < n and tries < n * 200:
        w = fn(rng)
        if w not in seen and min_len <= len(w) <= max_len:
            seen.add(w)
            out.append(w)
        tries += 1
    if len(out) < n:
        raise RuntimeError(f"Only generated {len(out)}/{n} unique words")
    return out


def _fmt(user_text: str, model_text: str) -> dict:
    text = (
        f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n{model_text}<end_of_turn>"
    )
    return {"text": text}


def _fmt_test(user_text: str, model_text: str, category: str) -> dict:
    d = _fmt(user_text, model_text)
    d["input"] = user_text
    d["expected"] = model_text
    d["category"] = category
    return d


def main() -> None:
    rng = random.Random(SEED)

    # --- vocabulary ---
    inputs  = _gen_unique(_make_fruit,  200, rng)
    outputs = _gen_unique(_make_output, 200, rng)
    mapping = dict(zip(inputs, outputs))  # input_word → output_word

    # --- singleton examples ---
    singletons = [_fmt(f"Map: {inp}", mapping[inp]) for inp in inputs]

    # --- composition pairs: pick 400 ordered pairs, split 350/50 ---
    all_pairs = list(itertools.combinations(inputs, 2))
    rng.shuffle(all_pairs)
    train_pairs = all_pairs[:350]
    heldout_pairs = all_pairs[350:400]

    train_comps = [
        _fmt(f"Map: {a} {b}", f"{mapping[a]} {mapping[b]}")
        for a, b in train_pairs
    ]
    heldout_comps = [
        _fmt_test(f"Map: {a} {b}", f"{mapping[a]} {mapping[b]}", "held_out_composition")
        for a, b in heldout_pairs
    ]

    # --- triple stretch pairs: 100 for training, 25 for test ---
    triple_set = list(itertools.combinations(inputs[:80], 3))
    rng.shuffle(triple_set)
    train_triple_comps = [
        _fmt(f"Map: {a} {b} {c}", f"{mapping[a]} {mapping[b]} {mapping[c]}")
        for a, b, c in triple_set[:100]
    ]
    triple_comps = [
        _fmt_test(
            f"Map: {a} {b} {c}",
            f"{mapping[a]} {mapping[b]} {mapping[c]}",
            "triple",
        )
        for a, b, c in triple_set[100:125]
    ]

    # --- singleton test samples (for eval completeness, not training) ---
    # We evaluate on the training singletons too (memorisation baseline)
    singleton_tests = [
        _fmt_test(f"Map: {inp}", mapping[inp], "singleton")
        for inp in rng.sample(inputs, 30)
    ]

    # --- train / valid split (90/10) ---
    train_pool = singletons + train_comps + train_triple_comps
    rng.shuffle(train_pool)
    split = int(len(train_pool) * 0.9)
    train_data  = train_pool[:split]
    valid_data  = train_pool[split:]

    test_data = singleton_tests + heldout_comps + triple_comps

    # --- write files ---
    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    write_jsonl(DATA_DIR / "train.jsonl", train_data)
    write_jsonl(DATA_DIR / "valid.jsonl", valid_data)
    write_jsonl(DATA_DIR / "test.jsonl",  test_data)

    print(f"train.jsonl  : {len(train_data)} examples")
    print(f"valid.jsonl  : {len(valid_data)} examples")
    print(f"test.jsonl   : {len(test_data)} examples")
    print(f"  singleton  : {len(singleton_tests)}")
    print(f"  held_out   : {len(heldout_comps)}")
    print(f"  triple     : {len(triple_comps)}")

    print("\n--- First 3 train examples ---")
    for ex in train_data[:3]:
        print(json.dumps(ex))

    print("\n--- First 3 test examples ---")
    for ex in test_data[:3]:
        print(json.dumps(ex))

    # Save mapping for eval reference
    with open(DATA_DIR / "mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping saved to {DATA_DIR}/mapping.json")


if __name__ == "__main__":
    main()
