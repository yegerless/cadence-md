"""Authentication request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """Payload for creating a new user account."""

    email: str = Field(description="User email address.", examples=["doctor@example.org"])
    password: str = Field(
        min_length=8,
        max_length=128,
        description="Plain password submitted over HTTPS.",
        examples=["correct-horse-battery-staple"],
    )
    first_name: str | None = Field(default=None, max_length=100)
    last_name: str | None = Field(default=None, max_length=100)


class LoginRequest(BaseModel):
    """Payload for password-based login."""

    email: str = Field(description="User email address.", examples=["doctor@example.org"])
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """Bearer token response returned after successful authentication."""

    access_token: str = Field(description="Short-lived API access token.")
    token_type: str = Field(default="bearer")
    expires_in: int = Field(gt=0, description="Access token lifetime in seconds.")
