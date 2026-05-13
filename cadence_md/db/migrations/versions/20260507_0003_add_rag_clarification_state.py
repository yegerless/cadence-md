"""Add RAG clarification wait state.

Revision ID: 20260507_0003
Revises: 20260506_0002
Create Date: 2026-05-07 19:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260507_0003"
down_revision: str | None = "20260506_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_RAG_REQUEST_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
NEW_RAG_REQUEST_STATUSES = (
    "queued",
    "running",
    "awaiting_clarification",
    "succeeded",
    "failed",
    "cancelled",
)


def upgrade() -> None:
    """Add clarification fields and allow the awaiting_clarification status."""
    with op.batch_alter_table("rag_requests") as batch_op:
        batch_op.drop_constraint("rag_request_status", type_="check")
        batch_op.add_column(sa.Column("clarification_question", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("clarification_answer", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("clarification_requested_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("clarification_answered_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "clarification_attempts",
                sa.Integer(),
                server_default=sa.text("0"),
                nullable=False,
            )
        )
        batch_op.create_check_constraint(
            "ck_rag_requests_rag_request_status",
            f"status IN {NEW_RAG_REQUEST_STATUSES!r}",
        )
        batch_op.create_check_constraint(
            "ck_rag_requests_clarification_attempts_non_negative",
            "clarification_attempts >= 0",
        )


def downgrade() -> None:
    """Remove clarification fields and restore the previous status set."""
    op.execute("UPDATE rag_requests SET status = 'failed' WHERE status = 'awaiting_clarification'")
    with op.batch_alter_table("rag_requests") as batch_op:
        batch_op.drop_constraint(
            "ck_rag_requests_clarification_attempts_non_negative",
            type_="check",
        )
        batch_op.drop_constraint("ck_rag_requests_rag_request_status", type_="check")
        batch_op.create_check_constraint(
            "ck_rag_requests_rag_request_status",
            f"status IN {OLD_RAG_REQUEST_STATUSES!r}",
        )
        batch_op.drop_column("clarification_attempts")
        batch_op.drop_column("clarification_answered_at")
        batch_op.drop_column("clarification_requested_at")
        batch_op.drop_column("clarification_answer")
        batch_op.drop_column("clarification_question")
