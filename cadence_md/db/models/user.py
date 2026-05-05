"""User ORM model."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from cadence_md.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from cadence_md.db.models.rag_request import RAGRequestLog


class User(TimestampMixin, Base):
    """Registered backend user."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    first_name: Mapped[str | None] = mapped_column(String(100))
    last_name: Mapped[str | None] = mapped_column(String(100))

    rag_requests: Mapped[list[RAGRequestLog]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
