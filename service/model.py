# ─────────────────────────────────────────────────────────────────────────────
# model.py — MLX inference backend (Apple Silicon / macOS only)
#
# Responsible for: loading the model + LoRA adapter, running inference
# (blocking and streaming), and token counting for usage metrics.
#
# TO DEPLOY ON A LINUX SERVER: see TRANSITION.md — only this file changes.
# Every other service file (main.py, auth.py, schemas.py …) stays identical.
# ─────────────────────────────────────────────────────────────────────────────
import logging
import sys
from pathlib import Path
from typing import Generator, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to reuse the shared Gemma prompt formatter from the training codebase.
# Falls back to an inline copy if the service is run outside the monorepo.
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from mlx_utils import format_gemma_prompt as _fmt_prompt
except ImportError:
    def _fmt_prompt(user_text: str, model_text=None) -> str:
        # Gemma 3 chat template: wraps user input in turn markers so the model
        # sees the exact same format it was trained on.
        return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

# Gemma appends this delimiter when it finishes its turn. We strip it so
# clients receive clean output without the model's internal control token.
_STOP_TOKEN = "<end_of_turn>"


class ModelState:
    """Holds the loaded model, tokenizer, and status flags.

    Stored on app.state so it is shared across all requests without globals.
    model_loaded / adapter_loaded are read by the /health endpoint.
    """
    def __init__(self):
        self.model = None           # mlx_lm model object (runs on Apple Metal)
        self.tokenizer = None       # HuggingFace-compatible tokenizer
        self.model_loaded: bool = False    # set True only after successful load
        self.adapter_loaded: bool = False  # set True only when LoRA adapter found


def _count_tokens(tokenizer, text: str) -> int:
    """Count tokens in text for usage reporting. Word-count fallback on error."""
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def _make_sampler(temperature: float):
    """Convert a temperature float to an mlx_lm sampler callable.

    mlx_lm's generate_step requires a `sampler` function — it does NOT accept
    a raw `temp` kwarg. make_sampler handles the conversion:
      temperature=0  → greedy argmax (deterministic, recommended for structured output)
      temperature>0  → categorical sampling with softmax temperature scaling
    """
    from mlx_lm.sample_utils import make_sampler
    return make_sampler(temp=temperature)


def load_model(model_path: Path, adapter_path: Optional[Path] = None) -> ModelState:
    """Load base model and optional LoRA adapter at startup.

    model_path accepts either a local directory or a HuggingFace repo ID
    (e.g. "mlx-community/gemma-3-1b-it-4bit") — mlx_lm handles both and
    will download + cache from HuggingFace if needed.

    If adapter_path doesn't exist on disk, the base model is loaded without
    it and adapter_loaded stays False (not an error — useful for testing).

    All exceptions are caught so the process stays alive and /health can
    report model_loaded: false instead of crashing on startup.
    """
    state = ModelState()
    try:
        from mlx_lm import load as mlx_load

        has_adapter = adapter_path is not None and adapter_path.exists()
        if has_adapter:
            logger.info("Loading model %s with adapter %s", model_path, adapter_path)
            state.model, state.tokenizer = mlx_load(
                str(model_path), adapter_path=str(adapter_path)
            )
            state.adapter_loaded = True
        else:
            if adapter_path is not None:
                logger.warning("Adapter path %s not found, loading base model only", adapter_path)
            logger.info("Loading base model from %s", model_path)
            state.model, state.tokenizer = mlx_load(str(model_path))

        state.model_loaded = True
        logger.info("Model ready")
    except Exception as exc:
        logger.error("Model load failed: %s", exc)

    return state


def run_inference(
    state: ModelState,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Tuple[str, int, int]:
    """Run a single blocking inference pass.

    Returns (output_text, prompt_token_count, completion_token_count).
    Called by the non-streaming path in /v1/chat/completions.
    """
    from mlx_lm import generate

    formatted = _fmt_prompt(prompt)
    prompt_tokens = _count_tokens(state.tokenizer, formatted)

    output: str = generate(
        state.model,
        state.tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=_make_sampler(temperature),
        verbose=False,
    )

    # Remove Gemma's closing turn delimiter if it appears in the output.
    if _STOP_TOKEN in output:
        output = output.split(_STOP_TOKEN)[0]
    return output.strip(), prompt_tokens, _count_tokens(state.tokenizer, output)


def run_inference_stream(
    state: ModelState,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    """Synchronous generator that yields text chunks as the model produces them.

    This is a blocking generator and must be run in a thread executor so the
    async event loop stays unblocked. See _sse_generator in main.py for how
    this is wired into the async SSE response.

    Stop-token boundary handling:
    stream_generate emits tokens in small variable-size chunks. The stop token
    "<end_of_turn>" (14 chars) can arrive split across two consecutive chunks
    (e.g. chunk N ends with "<end_of_" and chunk N+1 starts with "turn>").
    To handle this safely, we buffer the last (len(STOP)-1) characters before
    yielding, so we always have enough lookahead to detect the boundary.
    """
    from mlx_lm import stream_generate

    formatted = _fmt_prompt(prompt)
    buffer = ""

    for response in stream_generate(
        state.model,
        state.tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=_make_sampler(temperature),
    ):
        buffer += response.text

        # Stop token found: emit everything before it, then halt.
        if _STOP_TOKEN in buffer:
            safe = buffer.split(_STOP_TOKEN)[0]
            if safe:
                yield safe
            return

        # Emit only the "safe" prefix, holding back the last (len(STOP)-1)
        # chars in case the stop token starts here and completes next chunk.
        safe_len = max(0, len(buffer) - (len(_STOP_TOKEN) - 1))
        if safe_len > 0:
            yield buffer[:safe_len]
            buffer = buffer[safe_len:]

    # Flush any remaining buffered text after max_tokens is reached.
    if buffer.strip():
        yield buffer.strip()
