"""FastAPI application factory for the CADENCE-MD backend service."""

from __future__ import annotations

from fastapi import FastAPI

from cadence_md.backend.api.routers import api_router

API_V1_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    """Create the FastAPI app and register the public API contract routes."""
    app = FastAPI(
        title="CADENCE-MD Backend API",
        version="0.1.0",
        description="Versioned API contract for CADENCE-MD SPA and asynchronous RAG flow.",
    )
    app.include_router(api_router, prefix=API_V1_PREFIX)
    return app


app = create_app()
