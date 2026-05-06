"""Health check response schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class HealthStatus(StrEnum):
    """Common health status values."""

    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class HealthCheckDetail(BaseModel):
    """Structured detail for one dependency check without secret values."""

    status: HealthStatus
    message: str | None = None
    latency_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health response for process, dependency, and RAG readiness probes."""

    status: HealthStatus
    checks: dict[str, HealthStatus] = Field(default_factory=dict)
    details: dict[str, HealthCheckDetail] = Field(default_factory=dict)
