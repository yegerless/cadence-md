"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from cadence_md.backend.api.deps import get_backend_settings
from cadence_md.backend.exceptions import ApiError
from cadence_md.backend.limiter import limiter
from cadence_md.backend.request_context import get_route_settings
from cadence_md.backend.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from cadence_md.backend.schemas.errors import ErrorResponse
from cadence_md.backend.schemas.limits import AUTH_RATE_LIMIT
from cadence_md.backend.security import create_access_token, hash_password, verify_password
from cadence_md.backend.settings import BackendSettings
from cadence_md.db.repositories import UserRepository
from cadence_md.db.session import get_async_session

router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_RATE_LIMIT_RESPONSE = {
    429: {
        "model": ErrorResponse,
        "description": f"Rate limit exceeded: {AUTH_RATE_LIMIT}.",
    },
}

REGISTER_RESPONSES = {
    **AUTH_RATE_LIMIT_RESPONSE,
    409: {"model": ErrorResponse, "description": "Email already registered."},
}

LOGIN_RESPONSES = {
    **AUTH_RATE_LIMIT_RESPONSE,
    401: {"model": ErrorResponse, "description": "Invalid credentials."},
}


def _register_ip_limit(key: str) -> str:
    """SlowAPI passes ``key`` from ``key_func`` (client IP); limits come from request context."""
    return get_route_settings().AUTH_REGISTER_RATE_LIMIT_IP


def _login_ip_limit(key: str) -> str:
    return get_route_settings().AUTH_LOGIN_RATE_LIMIT_IP


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    responses=REGISTER_RESPONSES,
    summary="Register a new user",
    description=f"Creates a user account. Rate limit: {AUTH_RATE_LIMIT}.",
)
@limiter.limit(_register_ip_limit)
async def register_user(
    request: Request,
    body: RegisterRequest,
    session: AsyncSession = Depends(get_async_session),
    settings: BackendSettings = Depends(get_backend_settings),
) -> TokenResponse:
    """Register a user account."""
    email = body.email.strip().lower()
    repository = UserRepository(session)
    if await repository.get_by_email(email) is not None:
        raise ApiError(
            status_code=409,
            code="email_already_registered",
            message="Email already registered.",
        )
    user = await repository.create_user(
        email=email,
        password_hash=hash_password(body.password),
        first_name=body.first_name,
        last_name=body.last_name,
    )
    token, expires_in = create_access_token(user.id, settings)
    return TokenResponse(access_token=token, token_type="bearer", expires_in=expires_in)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    responses=LOGIN_RESPONSES,
    summary="Login with email and password",
    description=f"Authenticates a user and returns a bearer token. Rate limit: {AUTH_RATE_LIMIT}.",
)
@limiter.limit(_login_ip_limit)
async def login_user(
    request: Request,
    body: LoginRequest,
    session: AsyncSession = Depends(get_async_session),
    settings: BackendSettings = Depends(get_backend_settings),
) -> TokenResponse:
    """Authenticate a user and return an access token."""
    email = body.email.strip().lower()
    repository = UserRepository(session)
    user = await repository.get_by_email(email)
    if user is None or not verify_password(body.password, user.password_hash):
        raise ApiError(
            status_code=401,
            code="invalid_credentials",
            message="Invalid email or password.",
        )
    token, expires_in = create_access_token(user.id, settings)
    return TokenResponse(access_token=token, token_type="bearer", expires_in=expires_in)
