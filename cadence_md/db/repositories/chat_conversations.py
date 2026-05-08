"""Repository for chat conversation metadata."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cadence_md.db.models import ChatConversation, RAGRequestLog


class ChatConversationRepository:
    """Persistence operations for user-owned chat conversations."""

    def __init__(self, session: AsyncSession) -> None:
        """Store the session used by repository operations."""
        self._session = session

    async def create_chat(
        self,
        *,
        user_id: uuid.UUID,
        title: str | None = None,
    ) -> ChatConversation:
        """Create a chat conversation for a user."""
        chat = ChatConversation(user_id=user_id, title=title)
        self._session.add(chat)
        await self._session.flush()
        return chat

    async def get_active_chat_for_user(
        self,
        *,
        user_id: uuid.UUID,
        chat_id: uuid.UUID,
    ) -> ChatConversation | None:
        """Return a non-deleted chat only if it belongs to the given user."""
        result = await self._session.execute(
            select(ChatConversation).where(
                ChatConversation.id == chat_id,
                ChatConversation.user_id == user_id,
                ChatConversation.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    async def list_active_chats_for_user(
        self,
        *,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ChatConversation]:
        """Return active chats for a user, newest activity first."""
        result = await self._session.execute(
            select(ChatConversation)
            .where(
                ChatConversation.user_id == user_id,
                ChatConversation.deleted_at.is_(None),
            )
            .order_by(
                ChatConversation.last_message_at.desc().nullslast(),
                ChatConversation.updated_at.desc(),
                ChatConversation.created_at.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def soft_delete_chat_for_user(
        self,
        *,
        user_id: uuid.UUID,
        chat_id: uuid.UUID,
    ) -> ChatConversation | None:
        """Mark a chat as deleted without removing its messages."""
        result = await self._session.execute(
            select(ChatConversation).where(
                ChatConversation.id == chat_id,
                ChatConversation.user_id == user_id,
            )
        )
        chat = result.scalar_one_or_none()
        if chat is None:
            return None
        if chat.deleted_at is None:
            chat.deleted_at = datetime.now(UTC)
            await self._session.flush()
        return chat

    async def touch_chat_for_user(
        self,
        *,
        user_id: uuid.UUID,
        chat_id: uuid.UUID,
        title: str | None = None,
        last_message_at: datetime | None = None,
    ) -> ChatConversation | None:
        """Update chat activity metadata for a non-deleted chat."""
        chat = await self.get_active_chat_for_user(user_id=user_id, chat_id=chat_id)
        if chat is None:
            return None

        if title is not None:
            chat.title = title
        chat.last_message_at = last_message_at or datetime.now(UTC)
        await self._session.flush()
        return chat

    async def list_requests_by_chat_for_user(
        self,
        *,
        user_id: uuid.UUID,
        chat_id: uuid.UUID,
    ) -> list[RAGRequestLog]:
        """Return a user's RAG requests for a chat in chronological order."""
        result = await self._session.execute(
            select(RAGRequestLog)
            .where(
                RAGRequestLog.user_id == user_id,
                RAGRequestLog.chat_id == chat_id,
            )
            .order_by(RAGRequestLog.created_at.asc())
        )
        return list(result.scalars().all())
