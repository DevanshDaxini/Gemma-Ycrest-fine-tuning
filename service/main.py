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
    ReportListItem,
    ReportRequest,
    ReportResponse,
    ReportUsage,
    Usage,
    ValidateRequest,
    ValidateResponse,
)
from session_store import append_turn, clear_session, get_history, session_exists
from anonymizer import anonymize as _anonymize
from report_generator import generate_with_retry
from report_store import StoredReport, delete_report, get_report, list_reports, new_report_id, save_report
from prompt_template import list_templates, render as render_template
from html_renderer import render_report_html
from ndjson_validator import validate_ndjson

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


def _build_prompt(body: ChatCompletionRequest, user_content: str) -> tuple[str, bool]:
    """Build the prompt string and whether it is pre-formatted.

    user_content is passed explicitly so the caller can supply the anonymized
    version before this function prepends session history.

    If session_id is set and history exists, all prior turns are prepended using
    Gemma's multi-turn template and raw_prompt=True is returned so inference
    functions skip re-wrapping. Otherwise falls back to the single-turn path.

    Returns (prompt_string, raw_prompt_flag).
    """
    if body.session_id:
        history = get_history(body.session_id)
        if history:
            all_messages = history + [{"role": "user", "content": user_content}]
            return format_messages(all_messages), True

    return user_content, False


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

    # Anonymize before the content reaches the model, session store, or logs.
    # Controlled by ANONYMIZE=true in .env or environment. Off by default.
    if settings.anonymize:
        result = _anonymize(user_content)
        user_content = result.anonymized
        if result.mapping:
            logger.info("Anonymized %d entities before inference", len(result.mapping))

    prompt, raw_prompt = _build_prompt(body, user_content)
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


# ── Report generation ─────────────────────────────────────────────────────────

@app.post("/v1/reports", response_model=ReportResponse)
async def create_report(request: Request, body: ReportRequest):
    """Generate a structured NDJSON revenue brief with automatic retry on invalid output."""
    state = request.app.state.model_state
    if not state.model_loaded:
        return JSONResponse(status_code=503, content={"error": "model not loaded"})

    try:
        prompt_text = render_template(body.template, input=body.input)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    messages = [{"role": "user", "content": prompt_text}]
    loop = asyncio.get_running_loop()
    output, pt, ct, attempts, valid, errors = await loop.run_in_executor(
        None,
        lambda: generate_with_retry(
            state, messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            max_retries=body.max_retries,
        ),
    )

    report = StoredReport(
        id=new_report_id(),
        created_at=int(time.time()),
        input_text=body.input,
        template_name=body.template,
        output=output,
        valid=valid,
        errors=errors,
        attempts=attempts,
        prompt_tokens=pt,
        completion_tokens=ct,
    )
    save_report(report)

    if valid:
        logger.info("Report %s generated — %d attempts, %d tokens", report.id, attempts, pt + ct)
    else:
        logger.warning("Report %s INVALID after %d attempts", report.id, attempts)

    return ReportResponse(
        id=report.id,
        created_at=report.created_at,
        template=report.template_name,
        valid=valid,
        attempts=attempts,
        output=output,
        errors=errors,
        usage=ReportUsage(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
        ),
    )


@app.get("/v1/reports")
async def list_all_reports():
    """List all stored reports, newest first."""
    return [
        ReportListItem(
            id=r.id,
            created_at=r.created_at,
            template=r.template_name,
            valid=r.valid,
            attempts=r.attempts,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
        )
        for r in list_reports()
    ]


@app.get("/v1/reports/{report_id}", response_model=ReportResponse)
async def get_report_by_id(report_id: str):
    """Retrieve a stored report by ID."""
    report = get_report(report_id)
    if report is None:
        return JSONResponse(status_code=404, content={"error": "report not found"})
    return ReportResponse(
        id=report.id,
        created_at=report.created_at,
        template=report.template_name,
        valid=report.valid,
        attempts=report.attempts,
        output=report.output,
        errors=report.errors,
        usage=ReportUsage(
            prompt_tokens=report.prompt_tokens,
            completion_tokens=report.completion_tokens,
            total_tokens=report.prompt_tokens + report.completion_tokens,
        ),
    )


@app.delete("/v1/reports/{report_id}")
async def delete_report_by_id(report_id: str):
    """Delete a stored report. Returns 404 if not found."""
    if not delete_report(report_id):
        return JSONResponse(status_code=404, content={"error": "report not found"})
    return {"deleted": report_id}


@app.get("/v1/reports/{report_id}/html", response_class=None)
async def report_html(report_id: str):
    """Render a stored report as an HTML page for leadership dashboards."""
    from fastapi.responses import HTMLResponse
    report = get_report(report_id)
    if report is None:
        return JSONResponse(status_code=404, content={"error": "report not found"})

    _, _, parsed = validate_ndjson(report.output)
    html_content = render_report_html(
        report_id=report.id,
        created_at=report.created_at,
        template_name=report.template_name,
        valid=report.valid,
        attempts=report.attempts,
        records=parsed,
    )
    return HTMLResponse(content=html_content)


@app.get("/v1/templates")
async def list_available_templates():
    """List available prompt templates."""
    return {"templates": list_templates()}
