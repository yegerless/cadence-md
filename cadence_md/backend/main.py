"""FastAPI application factory for the CADENCE-MD backend service."""

from __future__ import annotations

from fastapi import FastAPI
from slowapi.middleware import SlowAPIMiddleware

from cadence_md.backend.api.routers import api_router
from cadence_md.backend.error_handlers import register_exception_handlers
from cadence_md.backend.limiter import limiter
from cadence_md.backend.request_context import RequestContextMiddleware
from cadence_md.backend.settings import backend_settings

API_V1_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    """Create the FastAPI app and register the public API contract routes."""
    app = FastAPI(
        title="CADENCE-MD Backend API",
        version="0.1.0",
        description="Versioned API contract for CADENCE-MD SPA and asynchronous RAG flow.",
    )
    app.state.settings = backend_settings
    app.state.limiter = limiter
    register_exception_handlers(app)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(api_router, prefix=API_V1_PREFIX)
    return app


app = create_app()
