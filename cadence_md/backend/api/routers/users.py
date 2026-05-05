"""User routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from cadence_md.backend.api.deps import get_current_user
from cadence_md.backend.schemas.errors import ErrorResponse
from cadence_md.backend.schemas.user import UserProfileResponse
from cadence_md.db.models import User

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
async def get_current_user_profile(
    user: User = Depends(get_current_user),
) -> UserProfileResponse:
    """Return the current authenticated user's profile."""
    return UserProfileResponse(
        id=str(user.id),
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
    )
