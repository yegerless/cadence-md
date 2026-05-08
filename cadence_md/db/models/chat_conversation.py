"""Chat conversation ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from cadence_md.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from cadence_md.db.models.rag_request import RAGRequestLog
    from cadence_md.db.models.user import User


class ChatConversation(TimestampMixin, Base):
    """Persisted chat thread metadata for a single user."""

    __tablename__ = "chat_conversations"
    __table_args__ = (
        Index(
            "ix_chat_conversations_user_deleted_last_message_updated",
            "user_id",
            "deleted_at",
            "last_message_at",
            "updated_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(String(255))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="chat_conversations")
    rag_requests: Mapped[list[RAGRequestLog]] = relationship(back_populates="chat")
