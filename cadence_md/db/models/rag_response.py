"""RAG response log ORM model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from cadence_md.db.base import Base, CreatedAtMixin

if TYPE_CHECKING:
    from cadence_md.db.models.rag_request import RAGRequestLog


class RAGResponseLog(CreatedAtMixin, Base):
    """Persisted RAG answer and telemetry for a completed request."""

    __tablename__ = "rag_responses"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rag_request_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("rag_requests.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[list[dict[str, object]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    latency_ms_json: Mapped[dict[str, float]] = mapped_column(JSON, default=dict, nullable=False)
    flags_json: Mapped[dict[str, bool]] = mapped_column(JSON, default=dict, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    langfuse_trace_id: Mapped[str | None] = mapped_column(String(255))

    rag_request: Mapped[RAGRequestLog] = relationship(back_populates="response")
