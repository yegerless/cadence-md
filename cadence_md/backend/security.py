"""Password hashing and JWT access tokens for the backend API."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from cadence_md.backend.exceptions import ApiError
from cadence_md.backend.settings import BackendSettings


def hash_password(plain: str) -> str:
    """Hash a plaintext password for storage (bcrypt)."""
    digest = bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt())
    return digest.decode("ascii")


def verify_password(plain: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("ascii"))


def create_access_token(user_id: uuid.UUID, settings: BackendSettings) -> tuple[str, int]:
    """Return ``(jwt_token, expires_in_seconds)`` for ``TokenResponse``."""
    now = datetime.now(tz=UTC)
    expire = now + timedelta(seconds=settings.JWT_ACCESS_TOKEN_EXPIRE_SECONDS)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "iat": now,
        "type": "access",
    }
    token = jwt.encode(
        payload,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    return token, settings.JWT_ACCESS_TOKEN_EXPIRE_SECONDS


def decode_access_token(token: str, settings: BackendSettings) -> uuid.UUID:
    """Decode a JWT access token and return the user id from ``sub``."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.PyJWTError:
        raise ApiError(
            status_code=401,
            code="invalid_token",
            message="Invalid or expired access token.",
        ) from None
    sub = payload.get("sub")
    if sub is None:
        raise ApiError(
            status_code=401,
            code="invalid_token",
            message="Invalid or expired access token.",
        )
    try:
        return uuid.UUID(str(sub))
    except ValueError as exc:
        raise ApiError(
            status_code=401,
            code="invalid_token",
            message="Invalid or expired access token.",
        ) from exc
