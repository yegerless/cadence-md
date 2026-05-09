"""Database enums shared by ORM models and repositories."""

from enum import StrEnum


class RAGRequestStatus(StrEnum):
    """Lifecycle statuses persisted for asynchronous RAG requests."""

    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
