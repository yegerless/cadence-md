"""Health route contracts."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from cadence_md.backend.api.deps import get_health_service
from cadence_md.backend.schemas.health import HealthResponse, HealthStatus
from cadence_md.backend.services.health import HealthService

router = APIRouter(prefix="/health", tags=["health"])

UNREADY_RESPONSES = {
    503: {
        "model": HealthResponse,
        "description": "One or more readiness dependencies are unavailable.",
    }
}


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
    responses=UNREADY_RESPONSES,
    summary="Readiness probe",
    description="Reports whether backend, queue, Qdrant, and inference dependencies are ready.",
)
async def health_ready(
    response: Response,
    health_service: HealthService = Depends(get_health_service),
) -> HealthResponse:
    """Return readiness for Docker healthchecks and traffic routing."""
    health = await health_service.ready()
    if health.status == HealthStatus.UNAVAILABLE:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health


@router.get(
    "/rag",
    response_model=HealthResponse,
    responses=UNREADY_RESPONSES,
    summary="RAG dependency health",
    description="Reports Qdrant, inference, and queue health without exposing secrets.",
)
async def health_rag(
    response: Response,
    health_service: HealthService = Depends(get_health_service),
) -> HealthResponse:
    """Return detailed RAG dependency readiness without constructing the full RAG stack."""
    health = await health_service.rag()
    if health.status == HealthStatus.UNAVAILABLE:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health
