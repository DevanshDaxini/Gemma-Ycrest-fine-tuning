# auth.py — service-to-service API key authentication
#
# Implemented as Starlette middleware so it runs before every route handler.
# Keys are loaded from the API_KEYS env var (comma-separated list) so they
# can be rotated without code changes — just update the env var and restart.

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

# Paths that skip authentication entirely. /health must be public so load
# balancers and monitoring can check the service without needing a key.
EXEMPT_PATHS = {"/health", "/docs", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validates Bearer token on every non-exempt request.

    Expected header:  Authorization: Bearer <api_key>
    Returns 401 if:   header is missing, scheme is not "Bearer", or key is unknown.
    """
    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})

        # Slice off "Bearer " prefix (7 chars) to get the raw token.
        token = auth_header[len("Bearer "):]
        if token not in settings.api_key_set:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})

        return await call_next(request)
