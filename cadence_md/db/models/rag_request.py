"""RAG request log ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from cadence_md.db.base import Base, CreatedAtMixin
from cadence_md.db.enums import RAGRequestStatus

if TYPE_CHECKING:
    from cadence_md.db.models.chat_conversation import ChatConversation
    from cadence_md.db.models.rag_response import RAGResponseLog
    from cadence_md.db.models.user import User


def _status_values(enum_type: type[RAGRequestStatus]) -> list[str]:
    """Return persisted enum values instead of Python member names."""
    return [status.value for status in enum_type]


class RAGRequestLog(CreatedAtMixin, Base):
    """Persisted lifecycle state for an asynchronous RAG request."""

    __tablename__ = "rag_requests"
    __table_args__ = (
        Index("ix_rag_requests_user_id", "user_id"),
        Index("ix_rag_requests_status", "status"),
        Index("ix_rag_requests_created_at", "created_at"),
        Index("ix_rag_requests_query_hash", "query_hash"),
        Index("ix_rag_requests_user_chat_created", "user_id", "chat_id", "created_at"),
        Index(
            "uq_rag_requests_user_id_idempotency_key",
            "user_id",
            "idempotency_key",
            unique=True,
            sqlite_where=text("idempotency_key IS NOT NULL"),
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
        CheckConstraint("retry_count >= 0", name="retry_count_non_negative"),
        CheckConstraint("clarification_attempts >= 0", name="clarification_attempts_non_negative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    conversation_id: Mapped[str | None] = mapped_column(String(128))
    chat_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_conversations.id", ondelete="SET NULL"),
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[RAGRequestStatus] = mapped_column(
        Enum(
            RAGRequestStatus,
            values_callable=_status_values,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            name="rag_request_status",
        ),
        default=RAGRequestStatus.QUEUED,
        nullable=False,
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255))
    original_request_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("rag_requests.id", ondelete="SET NULL"),
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(255))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    clarification_question: Mapped[str | None] = mapped_column(Text)
    clarification_answer: Mapped[str | None] = mapped_column(Text)
    clarification_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    clarification_answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    clarification_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="rag_requests")
    chat: Mapped[ChatConversation | None] = relationship(back_populates="rag_requests")
    response: Mapped[RAGResponseLog | None] = relationship(
        back_populates="rag_request",
        cascade="all, delete-orphan",
        uselist=False,
    )
    original_request: Mapped[RAGRequestLog | None] = relationship(
        remote_side=[id],
        back_populates="retry_requests",
    )
    retry_requests: Mapped[list[RAGRequestLog]] = relationship(back_populates="original_request")
