"""Health route contracts."""

from __future__ import annotations

from fastapi import APIRouter

from cadence_md.backend.schemas.health import HealthResponse, HealthStatus

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/live",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Reports whether the FastAPI process is alive.",
)
async def health_live() -> HealthResponse:
    """Return process liveness without external dependency checks."""
    return HealthResponse(status=HealthStatus.OK, checks={"backend": HealthStatus.OK})


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description="Readiness contract for future backend dependency checks.",
)
async def health_ready() -> HealthResponse:
    """Return readiness skeleton without opening DB/Redis connections."""
    return HealthResponse(status=HealthStatus.OK, checks={"backend": HealthStatus.OK})


@router.get(
    "/rag",
    response_model=HealthResponse,
    summary="RAG dependency health",
    description="Health contract for future Qdrant, worker queue, and inference checks.",
)
async def health_rag() -> HealthResponse:
    """Return RAG health skeleton without constructing RAG or inference clients."""
    return HealthResponse(
        status=HealthStatus.OK,
        checks={
            "qdrant": HealthStatus.OK,
            "queue": HealthStatus.OK,
            "inference": HealthStatus.OK,
        },
    )
