"""Error response schemas shared by backend API routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Stable error response shape for frontend and API clients."""

    code: str = Field(
        examples=["rate_limit_exceeded"],
        description="Machine-readable error code.",
    )
    message: str = Field(
        examples=["Too many requests."],
        description="Human-readable error summary.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured error details for client handling.",
    )
    request_id: str | None = Field(
        default=None,
        description="Correlation id for support and logs, when available.",
    )
