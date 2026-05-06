"""Injectable hook for enqueueing RAG work (Celery stub until worker integration)."""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

logger = logging.getLogger(__name__)


class RAGEnqueueService(Protocol):
    """Enqueue-side contract for dispatching a persisted ``rag_requests.id`` to workers."""

    async def enqueue(self, request_id: uuid.UUID) -> None:
        """Signal that ``request_id`` is ready to be processed asynchronously."""
        ...


class NoopRAGEnqueueService:
    """Default implementation that performs no network I/O (tests override via DI)."""

    async def enqueue(self, request_id: uuid.UUID) -> None:
        """Log at debug for observability; Celery integration replaces this class."""
        logger.debug("RAG enqueue noop: request_id=%s", request_id)
