"""Request-scoped context (correlation id + backend settings for rate limits)."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from cadence_md.backend.settings import BackendSettings, backend_settings
from cadence_md.observability.logging import log_context

settings_ctx: ContextVar[BackendSettings | None] = ContextVar(
    "backend_settings_ctx",
    default=None,
)


def get_route_settings() -> BackendSettings:
    """Return settings for the current request (fallback for non-HTTP contexts)."""
    resolved = settings_ctx.get()
    return resolved if resolved is not None else backend_settings


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach ``X-Request-ID`` and settings context for downstream code (e.g. SlowAPI)."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        header_value = request.headers.get("x-request-id")
        rid = header_value.strip() if header_value else str(uuid.uuid4())
        request.state.request_id = rid
        settings: BackendSettings = request.app.state.settings
        token = settings_ctx.set(settings)
        try:
            with log_context(request_id=rid, correlation_id=rid):
                response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            settings_ctx.reset(token)
