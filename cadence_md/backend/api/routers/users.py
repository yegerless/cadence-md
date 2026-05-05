"""User route contracts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from cadence_md.backend.schemas.errors import ErrorResponse
from cadence_md.backend.schemas.user import UserProfileResponse

router = APIRouter(prefix="/users", tags=["users"])

UNAUTHORIZED_RESPONSE = {
    401: {
        "model": ErrorResponse,
        "description": "Authentication is required.",
    },
}


@router.get(
    "/me",
    response_model=UserProfileResponse,
    responses=UNAUTHORIZED_RESPONSE,
    summary="Get current user profile",
    description="Returns the authenticated user's profile.",
)
async def get_current_user_profile() -> UserProfileResponse:
    """Return the current authenticated user's profile."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Current user lookup is not implemented yet.",
    )
