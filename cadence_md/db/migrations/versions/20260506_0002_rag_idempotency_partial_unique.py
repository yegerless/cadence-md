"""Partial unique index on (user_id, idempotency_key).

Revision ID: 20260506_0002
Revises: 20260505_0001
Create Date: 2026-05-06 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260506_0002"
down_revision: str | None = "20260505_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Replace non-unique idempotency index with partial composite unique index."""
    op.drop_index("ix_rag_requests_idempotency_key", table_name="rag_requests")
    op.create_index(
        "uq_rag_requests_user_id_idempotency_key",
        "rag_requests",
        ["user_id", "idempotency_key"],
        unique=True,
        sqlite_where=sa.text("idempotency_key IS NOT NULL"),
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    """Restore non-unique single-column idempotency index."""
    op.drop_index("uq_rag_requests_user_id_idempotency_key", table_name="rag_requests")
    op.create_index(
        "ix_rag_requests_idempotency_key",
        "rag_requests",
        ["idempotency_key"],
        unique=False,
    )
