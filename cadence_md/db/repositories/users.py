"""User repository."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cadence_md.db.models import User


class UserRepository:
    """Persistence operations for users."""

    def __init__(self, session: AsyncSession) -> None:
        """Store the session used by repository operations."""
        self._session = session

    async def create_user(
        self,
        *,
        email: str,
        password_hash: str,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> User:
        """Create a user and flush it to get a stable id."""
        user = User(
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        """Return a user by id."""
        return await self._session.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        """Return a user by email."""
        result = await self._session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
