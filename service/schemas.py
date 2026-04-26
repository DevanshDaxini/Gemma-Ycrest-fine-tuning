# schemas.py — Pydantic request/response models
#
# The /v1/chat/completions request and response shapes mirror the OpenAI Chat
# Completions API so that any OpenAI-compatible client can talk to this service
# without modification (just swap the base URL and API key).

from typing import List, Any, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: str     # "user", "assistant", or "system"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-revenue-brief"
    messages: List[Message]
    temperature: float = 0.0   # 0 = greedy/deterministic (recommended for structured output)
    max_tokens: int = 1024
    stream: bool = False        # set True to receive token-by-token SSE response
    session_id: Optional[str] = None  # if set, prior turns are prepended from session store


class MessageResponse(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: MessageResponse
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Non-streaming response — returned when stream=false (the default)."""
    id: str          # "chatcmpl-<uuid>"
    object: str = "chat.completion"
    created: int     # Unix timestamp
    model: str
    choices: List[Choice]
    usage: Usage


class ValidateRequest(BaseModel):
    content: str     # raw NDJSON string to validate


class ValidateResponse(BaseModel):
    valid: bool              # True only if zero errors found
    errors: List[str]        # one entry per error, includes line number
    parsed: List[Any]        # successfully parsed JSON objects from the input


# ── Report generation ─────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    input: str                          # weekly business data (free text or structured)
    template: str = "default"           # prompt template name (maps to prompt_templates/<name>.txt)
    max_retries: int = 3                # retry attempts if output is invalid NDJSON
    temperature: float = 0.0            # 0 = deterministic (recommended)
    max_tokens: int = 1500              # reports are typically 800-1500 tokens


class ReportUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ReportResponse(BaseModel):
    id: str                             # "report-<uuid>"
    created_at: int                     # Unix timestamp
    template: str
    valid: bool                         # True if output passed NDJSON validation
    attempts: int                       # how many inference attempts were made
    output: str                         # raw NDJSON string
    errors: List[str]                   # validation errors (empty if valid=True)
    usage: ReportUsage


class ReportListItem(BaseModel):
    id: str
    created_at: int
    template: str
    valid: bool
    attempts: int
    prompt_tokens: int
    completion_tokens: int
