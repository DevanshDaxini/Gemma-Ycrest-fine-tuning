# schemas.py — Pydantic request/response models
#
# The /v1/chat/completions request and response shapes mirror the OpenAI Chat
# Completions API so that any OpenAI-compatible client can talk to this service
# without modification (just swap the base URL and API key).

from typing import List, Any
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
