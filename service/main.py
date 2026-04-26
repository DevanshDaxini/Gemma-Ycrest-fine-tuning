import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from auth import APIKeyMiddleware
from config import settings
from model import ModelState, format_messages, load_model, run_inference, run_inference_stream
from ndjson_validator import validate_ndjson
from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    MessageResponse,
    Usage,
    ValidateRequest,
    ValidateResponse,
)
from session_store import append_turn, clear_session, get_history, session_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, elapsed_ms)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_state = load_model(settings.model_path, settings.adapter_path)
    yield


app = FastAPI(title="Gemma Revenue Brief API", lifespan=lifespan)

# Outermost → innermost: logging wraps auth wraps routes
app.add_middleware(APIKeyMiddleware)
app.add_middleware(LoggingMiddleware)


def _build_prompt(body: ChatCompletionRequest) -> tuple[str, bool]:
    """Build the prompt string and whether it is pre-formatted.

    If session_id is set and history exists, all prior turns are prepended using
    Gemma's multi-turn template and raw_prompt=True is returned so inference
    functions skip re-wrapping. Otherwise falls back to the single-turn path.

    Returns (prompt_string, raw_prompt_flag).
    """
    new_user_content = next(
        (m.content for m in reversed(body.messages) if m.role == "user"), ""
    )

    if body.session_id:
        history = get_history(body.session_id)
        if history:
            all_messages = history + [{"role": "user", "content": new_user_content}]
            return format_messages(all_messages), True

    return new_user_content, False


async def _sse_generator(
    state: ModelState,
    body: ChatCompletionRequest,
    prompt: str,
    raw_prompt: bool,
    user_content: str,
    completion_id: str,
    created: int,
):
    def _chunk(delta: dict, finish_reason=None) -> str:
        obj = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(obj)}\n\n"

    yield _chunk({"role": "assistant", "content": ""})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    collected: list[str] = []  # accumulate output for session storage

    def _produce():
        try:
            for text in run_inference_stream(
                state, prompt, body.temperature, body.max_tokens, raw_prompt=raw_prompt
            ):
                collected.append(text)
                loop.call_soon_threadsafe(queue.put_nowait, text)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    future = loop.run_in_executor(None, _produce)
    while (text := await queue.get()) is not None:
        yield _chunk({"content": text})

    await future

    # Store the completed turn in session history after streaming finishes.
    if body.session_id:
        append_turn(body.session_id, user_content, "".join(collected))

    yield _chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health(request: Request):
    state = request.app.state.model_state
    return {
        "status": "ok",
        "model_loaded": state.model_loaded,
        "adapter_loaded": state.adapter_loaded,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    state = request.app.state.model_state
    if not state.model_loaded:
        return JSONResponse(status_code=503, content={"error": "model not loaded"})

    # Extract the latest user message for session storage (before history is prepended).
    user_content = next(
        (m.content for m in reversed(body.messages) if m.role == "user"), ""
    )

    prompt, raw_prompt = _build_prompt(body)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if body.stream:
        return StreamingResponse(
            _sse_generator(state, body, prompt, raw_prompt, user_content, completion_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    output, prompt_tokens, completion_tokens = run_inference(
        state, prompt=prompt, temperature=body.temperature,
        max_tokens=body.max_tokens, raw_prompt=raw_prompt,
    )

    # Store turn after successful inference.
    if body.session_id:
        append_turn(body.session_id, user_content, output)

    if body.session_id:
        logger.info("session=%s turns stored", body.session_id)

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=body.model,
        choices=[
            Choice(
                index=0,
                message=MessageResponse(role="assistant", content=output),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear all stored history for a session. Returns 404 if session doesn't exist."""
    if not session_exists(session_id):
        return JSONResponse(status_code=404, content={"error": "session not found"})
    clear_session(session_id)
    return {"deleted": session_id}


@app.post("/v1/validate", response_model=ValidateResponse)
async def validate(body: ValidateRequest):
    valid, errors, parsed = validate_ndjson(body.content)
    return ValidateResponse(valid=valid, errors=errors, parsed=parsed)
