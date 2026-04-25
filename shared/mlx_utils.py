"""Shared MLX helpers for gemma fine-tune practice."""
from pathlib import Path


def format_gemma_prompt(user_text: str, model_text: str | None = None) -> str:
    """Gemma 3 chat template. Omit model_text for inference."""
    s = f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
    if model_text is not None:
        s += f"{model_text}<end_of_turn>"
    return s


def load_model_with_adapter(model_path: str, adapter_path: str | None = None):
    """Load MLX model + optional LoRA adapter."""
    from mlx_lm import load

    adapter = adapter_path if (adapter_path and Path(adapter_path).exists()) else None
    if adapter:
        model, tokenizer = load(model_path, adapter_path=adapter)
        print(f"[load] model={model_path}  adapter={adapter}")
    else:
        model, tokenizer = load(model_path)
        print(f"[load] model={model_path}  (no adapter)")
    return model, tokenizer


def greedy_generate(
    model, tokenizer, prompt: str, max_tokens: int = 256
) -> tuple[str, dict]:
    """
    Single greedy decode at temp=0.
    Returns (text, stats) where stats has input_tokens, output_tokens, elapsed_s, tokens_per_sec.
    """
    import time
    from mlx_lm import generate

    input_tokens = len(tokenizer.encode(prompt))
    t0 = time.perf_counter()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    elapsed = time.perf_counter() - t0

    if "<end_of_turn>" in out:
        out = out.split("<end_of_turn>")[0]
    out = out.strip()

    output_tokens = len(tokenizer.encode(out))
    stats = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": output_tokens / elapsed if elapsed > 0 else 0.0,
    }
    return out, stats


def distribution_generate(
    model, tokenizer, prompt: str, top_k: int = 5, n_tokens: int = 10
) -> list[list[tuple]]:
    """
    Greedy decode for n_tokens steps, capturing full top-k distribution each step.
    Returns list of steps; each step = [(token_str, prob, logprob), ...].

    Note: recomputes full attention each step (no KV cache) — fine for 10 tokens.
    """
    import mlx.core as mx
    import numpy as np

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Prepend BOS if needed
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos, *ids]

    x = mx.array(ids)[None]  # [1, seq_len]
    steps = []

    for _ in range(n_tokens):
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        last = logits[0, -1, :]
        mx.eval(last)

        lp = np.array(last.tolist(), dtype=np.float32)
        # Numerically stable log_softmax
        lp -= lp.max() + np.log(np.sum(np.exp(lp - lp.max())))

        top_idx = np.argsort(-lp)[:top_k]
        step = [(tokenizer.decode([int(i)]), float(np.exp(lp[i])), float(lp[i]))
                for i in top_idx]
        steps.append(step)

        next_tok = int(top_idx[0])
        x = mx.concatenate([x, mx.array([[next_tok]])], axis=1)

    return steps


def print_training_estimate(
    n_train: int,
    batch_size: int,
    iters: int,
    model_name: str,
    lora_rank: int = 8,
) -> None:
    """Print estimated training cost before starting."""
    steps_per_epoch = n_train / batch_size
    n_epochs = iters / steps_per_epoch if steps_per_epoch > 0 else 0
    secs_per_iter = 1.5  # empirical: Gemma 1B on M4, seq_len~256
    total_min = iters * secs_per_iter / 60
    mem_gb = 2.0 + batch_size * 0.35  # model fp16 + activations

    print("=" * 54)
    print(f"  Model     : {model_name}")
    print(f"  Train ex  : {n_train}  |  batch: {batch_size}")
    print(f"  Iters     : {iters}  (~{n_epochs:.1f} epochs)")
    print(f"  LoRA rank : {lora_rank}")
    print(f"  Est. time : ~{total_min:.0f} min on M4")
    print(f"  Est. mem  : ~{mem_gb:.1f} GB (fp16 base)")
    print("=" * 54)
