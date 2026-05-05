"""Health check response schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class HealthStatus(StrEnum):
    """Common health status values."""

    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class HealthResponse(BaseModel):
    """Lightweight health response without external service initialization."""

    status: HealthStatus
    checks: dict[str, HealthStatus] = Field(default_factory=dict)
