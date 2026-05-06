"""SlowAPI key functions for per-user chat rate limits."""

from __future__ import annotations

import jwt
from slowapi.util import get_remote_address
from starlette.requests import Request

from cadence_md.backend.request_context import get_route_settings


def chat_rate_limit_key(request: Request) -> str:
    """Prefer JWT ``sub`` for authenticated chat traffic; fall back to client IP."""
    settings = get_route_settings()
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
            )
            sub = payload.get("sub")
            if sub:
                return f"chat:user:{sub}"
        except jwt.PyJWTError:
            pass
    return f"chat:ip:{get_remote_address(request)}"
