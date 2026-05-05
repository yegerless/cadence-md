"""Create users and RAG log tables.

Revision ID: 20260505_0001
Revises:
Create Date: 2026-05-05 22:15:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260505_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

RAG_REQUEST_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")


def upgrade() -> None:
    """Create initial backend persistence schema."""
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("first_name", sa.String(length=100), nullable=True),
        sa.Column("last_name", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "rag_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.String(length=128), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("query_hash", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("original_request_id", sa.Uuid(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            f"status IN {RAG_REQUEST_STATUSES!r}",
            name=op.f("ck_rag_requests_rag_request_status"),
        ),
        sa.CheckConstraint(
            "retry_count >= 0",
            name=op.f("ck_rag_requests_retry_count_non_negative"),
        ),
        sa.ForeignKeyConstraint(["original_request_id"], ["rag_requests.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_requests")),
    )
    op.create_index("ix_rag_requests_created_at", "rag_requests", ["created_at"], unique=False)
    op.create_index(
        "ix_rag_requests_idempotency_key",
        "rag_requests",
        ["idempotency_key"],
        unique=False,
    )
    op.create_index("ix_rag_requests_query_hash", "rag_requests", ["query_hash"], unique=False)
    op.create_index("ix_rag_requests_status", "rag_requests", ["status"], unique=False)
    op.create_index("ix_rag_requests_user_id", "rag_requests", ["user_id"], unique=False)

    op.create_table(
        "rag_responses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("rag_request_id", sa.Uuid(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("sources_json", sa.JSON(), nullable=False),
        sa.Column("latency_ms_json", sa.JSON(), nullable=False),
        sa.Column("flags_json", sa.JSON(), nullable=False),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("langfuse_trace_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["rag_request_id"], ["rag_requests.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_responses")),
        sa.UniqueConstraint("rag_request_id", name=op.f("uq_rag_responses_rag_request_id")),
    )


def downgrade() -> None:
    """Drop initial backend persistence schema."""
    op.drop_table("rag_responses")
    op.drop_index("ix_rag_requests_user_id", table_name="rag_requests")
    op.drop_index("ix_rag_requests_status", table_name="rag_requests")
    op.drop_index("ix_rag_requests_query_hash", table_name="rag_requests")
    op.drop_index("ix_rag_requests_idempotency_key", table_name="rag_requests")
    op.drop_index("ix_rag_requests_created_at", table_name="rag_requests")
    op.drop_table("rag_requests")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
