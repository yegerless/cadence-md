"""FastAPI Prometheus instrumentation."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from cadence_md.observability.metrics import (
    observe_backend_request,
    prometheus_payload,
)

metrics_router = APIRouter(include_in_schema=False)


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path or request.url.path)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Collect request count, latency, and error metrics for FastAPI."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        if request.url.path == "/metrics":
            return await call_next(request)

        started = time.perf_counter()
        error_type: str | None = None
        try:
            response = await call_next(request)
        except Exception as exc:
            error_type = type(exc).__name__
            observe_backend_request(
                method=request.method,
                path=_route_path(request),
                status_code=500,
                duration_seconds=time.perf_counter() - started,
                error_type=error_type,
            )
            raise

        observe_backend_request(
            method=request.method,
            path=_route_path(request),
            status_code=response.status_code,
            duration_seconds=time.perf_counter() - started,
            error_type=error_type,
        )
        return response


@metrics_router.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics for the backend process."""
    return Response(content=prometheus_payload(), media_type=CONTENT_TYPE_LATEST)
