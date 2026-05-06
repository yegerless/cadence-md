"""FastAPI dependencies for backend routes."""

from __future__ import annotations

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from cadence_md.backend.exceptions import ApiError
from cadence_md.backend.security import decode_access_token
from cadence_md.backend.services.rag_enqueue import CeleryRAGEnqueueService, RAGEnqueueService
from cadence_md.backend.settings import BackendSettings, backend_settings
from cadence_md.db.models import User
from cadence_md.db.repositories import UserRepository
from cadence_md.db.session import get_async_session

http_bearer = HTTPBearer(auto_error=False)


def get_rag_enqueue() -> RAGEnqueueService:
    """Return the queue enqueue hook for asynchronous RAG processing."""
    return CeleryRAGEnqueueService()


def get_backend_settings() -> BackendSettings:
    """Return backend settings (override in tests via ``dependency_overrides``)."""
    return backend_settings


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
    session: AsyncSession = Depends(get_async_session),
    settings: BackendSettings = Depends(get_backend_settings),
) -> User:
    """Load the authenticated user from a Bearer JWT."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise ApiError(
            status_code=401,
            code="not_authenticated",
            message="Authentication is required.",
        )
    user_id = decode_access_token(credentials.credentials, settings)
    repository = UserRepository(session)
    user = await repository.get_by_id(user_id)
    if user is None:
        raise ApiError(
            status_code=401,
            code="not_authenticated",
            message="Authentication is required.",
        )
    return user
