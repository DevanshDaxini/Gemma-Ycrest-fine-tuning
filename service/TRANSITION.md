# Deploying on a Linux Server (Transitioning from MLX)

## Why MLX doesn't work on a server

MLX uses Apple's **Metal** GPU API, which only exists on macOS + Apple Silicon (M1/M2/M3/M4).
Linux servers run NVIDIA GPUs (CUDA) or CPU — Metal doesn't exist there, so `mlx_lm` won't import.

**The good news: only `model.py` changes.** Every other file — `main.py`, `auth.py`, `schemas.py`,
`ndjson_validator.py`, `config.py` — is backend-agnostic and stays identical.

---

## Step 1 — Fuse the LoRA adapter into the base model (on your Mac)

During training, LoRA stores only the *weight deltas* as a separate adapter file.
To deploy on a server, bake those deltas into the base model weights to produce a single
portable model in standard HuggingFace safetensors format.

```bash
# Run from the project root on your Mac
python -m mlx_lm.fuse \
  --model mlx-community/gemma-3-1b-it-4bit \
  --adapter-path ./phase2_ndjson/adapters \
  --save-path ./fused_model \
  --de-quantize
```

`--de-quantize` converts from 4-bit MLX format back to float16 safetensors.
The output in `./fused_model/` is a standard HuggingFace model directory.

---

## Step 2 — Move the model to your server

**Option A — HuggingFace Hub (recommended):**
```bash
huggingface-cli login
huggingface-cli upload your-org/gemma-revenue-brief ./fused_model --private
```
On the server, set `MODEL_PATH=your-org/gemma-revenue-brief` and the backend will
auto-download it.

**Option B — direct file copy:**
```bash
rsync -avz ./fused_model/ user@your-server:/opt/models/gemma-revenue-brief/
```
On the server, set `MODEL_PATH=/opt/models/gemma-revenue-brief`.

---

## Step 3 — Swap requirements.txt

Remove `mlx-lm`. Add your target backend.

**NVIDIA GPU server (recommended for throughput):**
```
fastapi>=0.110.0
uvicorn>=0.29.0
vllm>=0.4.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
```

**CPU-only or any hardware (more portable, slower):**
```
fastapi>=0.110.0
uvicorn>=0.29.0
transformers>=4.40.0
torch>=2.2.0
accelerate>=0.29.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
```

---

## Step 4 — Replace model.py

Copy one of the two drop-in replacements below over `service/model.py`.
Everything else stays untouched.

---

### Option A: vLLM backend (NVIDIA GPU — best throughput)

vLLM uses continuous batching and paged attention for high-throughput serving.
Use this if your server has an NVIDIA GPU.

```python
# model.py — vLLM backend (NVIDIA GPU / Linux server)
# Requires: pip install vllm>=0.4.0
import logging
import sys
from pathlib import Path
from typing import Generator, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from mlx_utils import format_gemma_prompt as _fmt_prompt
except ImportError:
    def _fmt_prompt(user_text: str, model_text=None) -> str:
        return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

_STOP_TOKEN = "<end_of_turn>"


class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded: bool = False
        self.adapter_loaded: bool = False


def _count_tokens(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def load_model(model_path: Path, adapter_path: Optional[Path] = None) -> ModelState:
    # adapter_path is unused here — the LoRA adapter should already be fused
    # into the model weights before deployment (see Step 1 in TRANSITION.md).
    state = ModelState()
    try:
        from vllm import LLM
        from transformers import AutoTokenizer

        logger.info("Loading model with vLLM from %s", model_path)
        state.model = LLM(model=str(model_path), dtype="float16")
        # Load tokenizer separately for token counting.
        state.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
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
    from vllm import SamplingParams

    formatted = _fmt_prompt(prompt)
    prompt_tokens = _count_tokens(state.tokenizer, formatted)

    # Pass the stop token directly to vLLM — it handles boundary detection internally.
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                             stop=[_STOP_TOKEN])
    outputs = state.model.generate([formatted], params)
    output = outputs[0].outputs[0].text.strip()

    return output, prompt_tokens, _count_tokens(state.tokenizer, output)


def run_inference_stream(
    state: ModelState,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    # vLLM's synchronous LLM.generate() doesn't support token-by-token streaming.
    # For true streaming, replace LLM with AsyncLLMEngine and update main.py
    # to call engine.generate() with streaming=True.
    # This implementation returns the full output in one chunk (still correct,
    # just not token-by-token).
    from vllm import SamplingParams

    formatted = _fmt_prompt(prompt)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                             stop=[_STOP_TOKEN])
    outputs = state.model.generate([formatted], params)
    output = outputs[0].outputs[0].text.strip()
    if output:
        yield output
```

---

### Option B: HuggingFace Transformers backend (CPU or any GPU)

Works on CPU, NVIDIA, AMD, and even Apple Silicon.
Slower than vLLM but more portable.

```python
# model.py — HuggingFace Transformers backend
# Requires: pip install transformers torch accelerate
import logging
import sys
from pathlib import Path
from threading import Thread
from typing import Generator, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from mlx_utils import format_gemma_prompt as _fmt_prompt
except ImportError:
    def _fmt_prompt(user_text: str, model_text=None) -> str:
        return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

_STOP_TOKEN = "<end_of_turn>"


class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded: bool = False
        self.adapter_loaded: bool = False


def _count_tokens(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


def load_model(model_path: Path, adapter_path: Optional[Path] = None) -> ModelState:
    state = ModelState()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model with transformers from %s", model_path)
        state.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        state.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",  # auto-places layers across available GPUs, falls back to CPU
        )
        state.model.eval()
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
    import torch

    formatted = _fmt_prompt(prompt)
    inputs = state.tokenizer(formatted, return_tensors="pt").to(state.model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = state.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=state.tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens — slice off the prompt prefix.
    new_tokens = output_ids[0][prompt_tokens:]
    output = state.tokenizer.decode(new_tokens, skip_special_tokens=True)

    if _STOP_TOKEN in output:
        output = output.split(_STOP_TOKEN)[0]
    return output.strip(), prompt_tokens, len(new_tokens)


def run_inference_stream(
    state: ModelState,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    import torch
    from transformers import TextIteratorStreamer

    formatted = _fmt_prompt(prompt)
    inputs = state.tokenizer(formatted, return_tensors="pt").to(state.model.device)

    # TextIteratorStreamer runs model.generate() in a background thread and
    # exposes the decoded tokens as an iterable — exactly what run_inference_stream
    # needs to be a synchronous generator.
    streamer = TextIteratorStreamer(
        state.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    thread = Thread(
        target=state.model.generate,
        kwargs={
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "pad_token_id": state.tokenizer.eos_token_id,
            "streamer": streamer,
        },
    )
    thread.start()

    # Apply the same sliding-window stop-token detection as the MLX backend.
    buffer = ""
    for text in streamer:
        buffer += text
        if _STOP_TOKEN in buffer:
            safe = buffer.split(_STOP_TOKEN)[0]
            if safe:
                yield safe
            thread.join()
            return
        safe_len = max(0, len(buffer) - (len(_STOP_TOKEN) - 1))
        if safe_len > 0:
            yield buffer[:safe_len]
            buffer = buffer[safe_len:]

    thread.join()
    if buffer.strip():
        yield buffer.strip()
```

---

## What changes and what doesn't

| File | Mac (MLX) | Linux server |
|---|---|---|
| `model.py` | **Replace** with vLLM or transformers version above | ← this |
| `requirements.txt` | **Replace** `mlx-lm` with `vllm` or `transformers torch accelerate` | ← this |
| `main.py` | No change | No change |
| `auth.py` | No change | No change |
| `config.py` | No change (`MODEL_PATH` env var still works) | No change |
| `schemas.py` | No change | No change |
| `ndjson_validator.py` | No change | No change |
| `Dockerfile` | Swap base image to `nvidia/cuda:12.1.0-runtime-ubuntu22.04` if GPU | optional |

## Choosing between vLLM and Transformers

| | vLLM | Transformers |
|---|---|---|
| Hardware | NVIDIA GPU required | CPU, NVIDIA, AMD, Apple Silicon |
| Throughput | High (continuous batching) | Lower |
| Token streaming | Needs AsyncLLMEngine for true streaming | Built-in via TextIteratorStreamer |
| Setup complexity | Higher | Lower |
| Best for | Production with concurrent users | Single-user, dev, or CPU-only servers |
