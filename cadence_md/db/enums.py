"""Database enums shared by ORM models and repositories."""

from enum import StrEnum


class RAGRequestStatus(StrEnum):
    """Lifecycle statuses persisted for asynchronous RAG requests."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
