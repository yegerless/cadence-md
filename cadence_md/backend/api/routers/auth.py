"""Authentication route contracts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from cadence_md.backend.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from cadence_md.backend.schemas.errors import ErrorResponse
from cadence_md.backend.schemas.limits import AUTH_RATE_LIMIT

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_RATE_LIMIT_RESPONSE = {
    429: {
        "model": ErrorResponse,
        "description": f"Rate limit exceeded: {AUTH_RATE_LIMIT}.",
    },
}


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    responses=AUTH_RATE_LIMIT_RESPONSE,
    summary="Register a new user",
    description=f"Creates a user account. Rate limit: {AUTH_RATE_LIMIT}.",
)
async def register_user(_: RegisterRequest) -> TokenResponse:
    """Register a user account."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Registration business logic is not implemented yet.",
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    responses=AUTH_RATE_LIMIT_RESPONSE,
    summary="Login with email and password",
    description=f"Authenticates a user and returns a bearer token. Rate limit: {AUTH_RATE_LIMIT}.",
)
async def login_user(_: LoginRequest) -> TokenResponse:
    """Authenticate a user and return an access token."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Login business logic is not implemented yet.",
    )
