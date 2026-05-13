"""User-facing profile schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfileResponse(BaseModel):
    """Current authenticated user's profile."""

    id: str = Field(description="Stable user identifier.")
    email: str = Field(description="User email address.")
    first_name: str | None = None
    last_name: str | None = None
