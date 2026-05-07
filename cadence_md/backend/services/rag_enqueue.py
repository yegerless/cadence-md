"""Injectable hook for enqueueing RAG work (Celery stub until worker integration)."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol

from cadence_md.backend.settings import BackendSettings, backend_settings
from cadence_md.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def make_rag_task_id(request_id: uuid.UUID, *, clarification_attempt: int | None = None) -> str:
    """Build a deterministic Celery task id for a persisted RAG request."""
    if clarification_attempt is not None and clarification_attempt > 0:
        return f"rag-request-{request_id}-clarification-{clarification_attempt}"
    return f"rag-request-{request_id}"


class RAGEnqueueService(Protocol):
    """Enqueue-side contract for dispatching a persisted ``rag_requests.id`` to workers."""

    async def enqueue(self, request_id: uuid.UUID, *, task_id: str | None = None) -> str:
        """Signal that ``request_id`` is ready to be processed asynchronously."""
        ...


class CeleryRAGEnqueueService:
    """Celery-backed implementation for dispatching RAG worker tasks."""

    def __init__(
        self,
        *,
        settings: BackendSettings = backend_settings,
        app: Any = celery_app,
    ) -> None:
        """Store Celery dependencies; tests may pass a lightweight fake app."""
        self._settings = settings
        self._app = app

    async def enqueue(self, request_id: uuid.UUID, *, task_id: str | None = None) -> str:
        """Send the RAG task to Celery and return its deterministic task id."""
        task_id = task_id or make_rag_task_id(request_id)
        self._app.send_task(
            self._settings.CELERY_RAG_TASK_NAME,
            args=(str(request_id),),
            task_id=task_id,
            queue=self._settings.CELERY_RAG_QUEUE_NAME,
        )
        logger.info(
            "RAG request enqueued",
            extra={"rag_request_id": str(request_id), "celery_task_id": task_id},
        )
        return task_id


class NoopRAGEnqueueService:
    """Default implementation that performs no network I/O (tests override via DI)."""

    async def enqueue(self, request_id: uuid.UUID, *, task_id: str | None = None) -> str:
        """Log at debug for observability; Celery integration replaces this class."""
        task_id = task_id or make_rag_task_id(request_id)
        logger.debug("RAG enqueue noop: request_id=%s task_id=%s", request_id, task_id)
        return task_id
