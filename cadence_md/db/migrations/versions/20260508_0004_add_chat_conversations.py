"""Add chat conversations.

Revision ID: 20260508_0004
Revises: 20260507_0003
Create Date: 2026-05-08 17:35:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260508_0004"
down_revision: str | None = "20260507_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create chat metadata and link RAG requests to chats."""
    op.create_table(
        "chat_conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_conversations")),
    )
    op.create_index(
        "ix_chat_conversations_user_deleted_last_message_updated",
        "chat_conversations",
        ["user_id", "deleted_at", "last_message_at", "updated_at"],
        unique=False,
    )

    # Existing legacy conversation_id values are intentionally not backfilled here:
    # chat_id remains nullable until the API starts creating ChatConversation rows.
    with op.batch_alter_table("rag_requests") as batch_op:
        batch_op.add_column(sa.Column("chat_id", sa.Uuid(), nullable=True))
        batch_op.create_foreign_key(
            op.f("fk_rag_requests_chat_id_chat_conversations"),
            "chat_conversations",
            ["chat_id"],
            ["id"],
            ondelete="SET NULL",
        )

    op.create_index(
        "ix_rag_requests_user_chat_created",
        "rag_requests",
        ["user_id", "chat_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove chat metadata and request links."""
    op.drop_index("ix_rag_requests_user_chat_created", table_name="rag_requests")
    with op.batch_alter_table("rag_requests") as batch_op:
        batch_op.drop_constraint(
            op.f("fk_rag_requests_chat_id_chat_conversations"),
            type_="foreignkey",
        )
        batch_op.drop_column("chat_id")

    op.drop_index(
        "ix_chat_conversations_user_deleted_last_message_updated",
        table_name="chat_conversations",
    )
    op.drop_table("chat_conversations")
