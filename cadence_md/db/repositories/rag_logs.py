"""Repositories for RAG request and response logs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cadence_md.db.enums import RAGRequestStatus
from cadence_md.db.models import RAGRequestLog, RAGResponseLog


class DuplicateRAGResponseError(ValueError):
    """Raised when a response already exists for a RAG request."""


class RAGLogRepository:
    """Persistence operations for asynchronous RAG request lifecycle logs."""

    def __init__(self, session: AsyncSession) -> None:
        """Store the session used by repository operations."""
        self._session = session

    async def create_request(
        self,
        *,
        user_id: uuid.UUID,
        query: str,
        query_hash: str,
        conversation_id: str | None = None,
        celery_task_id: str | None = None,
        original_request_id: uuid.UUID | None = None,
        idempotency_key: str | None = None,
        retry_count: int = 0,
    ) -> RAGRequestLog:
        """Create a queued RAG request log."""
        request_log = RAGRequestLog(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            query_hash=query_hash,
            status=RAGRequestStatus.QUEUED,
            celery_task_id=celery_task_id,
            original_request_id=original_request_id,
            idempotency_key=idempotency_key,
            retry_count=retry_count,
        )
        self._session.add(request_log)
        await self._session.flush()
        return request_log

    async def create_retry_request(
        self,
        *,
        original_request_id: uuid.UUID,
        idempotency_key: str | None = None,
        celery_task_id: str | None = None,
    ) -> RAGRequestLog:
        """Create a queued retry request linked to an existing request."""
        original = await self.get_request(original_request_id)
        if original is None:
            msg = f"Original RAG request was not found: {original_request_id}"
            raise ValueError(msg)

        original.retry_count += 1
        retry_request = await self.create_request(
            user_id=original.user_id,
            query=original.query,
            query_hash=original.query_hash,
            conversation_id=original.conversation_id,
            celery_task_id=celery_task_id,
            original_request_id=original.id,
            idempotency_key=idempotency_key,
            retry_count=original.retry_count,
        )
        await self._session.flush()
        return retry_request

    async def get_request(self, request_id: uuid.UUID) -> RAGRequestLog | None:
        """Return a RAG request by id."""
        return await self._session.get(RAGRequestLog, request_id)

    async def get_request_for_user(
        self,
        *,
        request_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> RAGRequestLog | None:
        """Return a RAG request only if it belongs to the given user."""
        result = await self._session.execute(
            select(RAGRequestLog).where(
                RAGRequestLog.id == request_id,
                RAGRequestLog.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def count_active_requests(self) -> int:
        """Count queued or running requests (global queue depth)."""
        stmt = (
            select(func.count())
            .select_from(RAGRequestLog)
            .where(
                RAGRequestLog.status.in_((RAGRequestStatus.QUEUED, RAGRequestStatus.RUNNING)),
            )
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def get_request_by_idempotency_key(
        self,
        *,
        user_id: uuid.UUID,
        idempotency_key: str,
    ) -> RAGRequestLog | None:
        """Return the request for this user and idempotency key, if any."""
        result = await self._session.execute(
            select(RAGRequestLog).where(
                RAGRequestLog.user_id == user_id,
                RAGRequestLog.idempotency_key == idempotency_key,
            )
        )
        return result.scalar_one_or_none()

    async def mark_running(
        self,
        request_id: uuid.UUID,
        *,
        celery_task_id: str | None = None,
    ) -> RAGRequestLog | None:
        """Mark a request as running and store its start timestamp."""
        return await self._update_request(
            request_id,
            status=RAGRequestStatus.RUNNING,
            started_at=datetime.now(UTC),
            celery_task_id=celery_task_id,
        )

    async def claim_queued_request(
        self,
        request_id: uuid.UUID,
        *,
        celery_task_id: str | None = None,
    ) -> RAGRequestLog | None:
        """Atomically move a queued request to running and return it if claimed."""
        return await self._guarded_status_update(
            request_id,
            from_status=RAGRequestStatus.QUEUED,
            to_status=RAGRequestStatus.RUNNING,
            started_at=datetime.now(UTC),
            celery_task_id=celery_task_id,
        )

    async def mark_succeeded(self, request_id: uuid.UUID) -> RAGRequestLog | None:
        """Mark a request as succeeded and store its finish timestamp."""
        return await self._update_request(
            request_id,
            status=RAGRequestStatus.SUCCEEDED,
            finished_at=datetime.now(UTC),
        )

    async def mark_succeeded_from_running(self, request_id: uuid.UUID) -> RAGRequestLog | None:
        """Atomically mark a running request as succeeded."""
        return await self._guarded_status_update(
            request_id,
            from_status=RAGRequestStatus.RUNNING,
            to_status=RAGRequestStatus.SUCCEEDED,
            finished_at=datetime.now(UTC),
        )

    async def mark_failed(self, request_id: uuid.UUID) -> RAGRequestLog | None:
        """Mark a request as failed and store its finish timestamp."""
        return await self._update_request(
            request_id,
            status=RAGRequestStatus.FAILED,
            finished_at=datetime.now(UTC),
        )

    async def mark_failed_from_running(self, request_id: uuid.UUID) -> RAGRequestLog | None:
        """Atomically mark a running request as failed."""
        return await self._guarded_status_update(
            request_id,
            from_status=RAGRequestStatus.RUNNING,
            to_status=RAGRequestStatus.FAILED,
            finished_at=datetime.now(UTC),
        )

    async def mark_cancelled(self, request_id: uuid.UUID) -> RAGRequestLog | None:
        """Mark a request as cancelled and store its finish timestamp."""
        return await self._update_request(
            request_id,
            status=RAGRequestStatus.CANCELLED,
            finished_at=datetime.now(UTC),
        )

    async def mark_cancelled_from_running(self, request_id: uuid.UUID) -> RAGRequestLog | None:
        """Atomically mark a running request as cancelled."""
        return await self._guarded_status_update(
            request_id,
            from_status=RAGRequestStatus.RUNNING,
            to_status=RAGRequestStatus.CANCELLED,
            finished_at=datetime.now(UTC),
        )

    async def request_cancel(self, request_id: uuid.UUID) -> RAGRequestLog | None:
        """Record that cancellation was requested for a queued or running request."""
        return await self._update_request(request_id, cancel_requested_at=datetime.now(UTC))

    async def set_queued_task_id(
        self,
        request_id: uuid.UUID,
        *,
        celery_task_id: str,
    ) -> RAGRequestLog | None:
        """Store a Celery task id only while the request is still queued."""
        return await self._guarded_update(
            request_id,
            allowed_statuses=(RAGRequestStatus.QUEUED,),
            celery_task_id=celery_task_id,
        )

    async def increment_retry(self, request_id: uuid.UUID) -> RAGRequestLog | None:
        """Increment retry count on an existing request."""
        request_log = await self.get_request(request_id)
        if request_log is None:
            return None

        request_log.retry_count += 1
        await self._session.flush()
        return request_log

    async def create_response(
        self,
        *,
        rag_request_id: uuid.UUID,
        answer: str,
        sources: list[dict[str, Any]] | None = None,
        latency_ms: dict[str, float] | None = None,
        flags: dict[str, bool] | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        langfuse_trace_id: str | None = None,
    ) -> RAGResponseLog:
        """Create a response log, guarding against duplicate results."""
        existing_response = await self.get_response_for_request(rag_request_id)
        if existing_response is not None:
            msg = f"RAG response already exists for request: {rag_request_id}"
            raise DuplicateRAGResponseError(msg)

        response_log = RAGResponseLog(
            rag_request_id=rag_request_id,
            answer=answer,
            sources_json=sources or [],
            latency_ms_json=latency_ms or {},
            flags_json=flags or {},
            error_type=error_type,
            error_message=error_message,
            langfuse_trace_id=langfuse_trace_id,
        )
        self._session.add(response_log)
        await self._session.flush()
        return response_log

    async def get_response_for_request(self, rag_request_id: uuid.UUID) -> RAGResponseLog | None:
        """Return the response associated with a RAG request."""
        result = await self._session.execute(
            select(RAGResponseLog).where(RAGResponseLog.rag_request_id == rag_request_id)
        )
        return result.scalar_one_or_none()

    async def has_response_for_request(self, rag_request_id: uuid.UUID) -> bool:
        """Return whether a response already exists for a RAG request."""
        result = await self._session.execute(
            select(func.count())
            .select_from(RAGResponseLog)
            .where(RAGResponseLog.rag_request_id == rag_request_id)
        )
        return int(result.scalar_one()) > 0

    async def _update_request(
        self,
        request_id: uuid.UUID,
        **values: object,
    ) -> RAGRequestLog | None:
        """Load a request, assign values, and flush the session."""
        request_log = await self.get_request(request_id)
        if request_log is None:
            return None

        for field_name, value in values.items():
            if value is not None:
                setattr(request_log, field_name, value)
        await self._session.flush()
        return request_log

    async def _guarded_status_update(
        self,
        request_id: uuid.UUID,
        *,
        from_status: RAGRequestStatus,
        to_status: RAGRequestStatus,
        **values: object,
    ) -> RAGRequestLog | None:
        """Update a request only if it is currently in the expected status."""
        return await self._guarded_update(
            request_id,
            allowed_statuses=(from_status,),
            status=to_status,
            **values,
        )

    async def _guarded_update(
        self,
        request_id: uuid.UUID,
        *,
        allowed_statuses: tuple[RAGRequestStatus, ...],
        **values: object,
    ) -> RAGRequestLog | None:
        """Run an atomic status-guarded UPDATE and reload the changed row."""
        update_values = {
            field_name: value for field_name, value in values.items() if value is not None
        }
        if not update_values:
            return await self.get_request(request_id)

        stmt = (
            update(RAGRequestLog)
            .where(
                RAGRequestLog.id == request_id,
                RAGRequestLog.status.in_(allowed_statuses),
            )
            .values(**update_values)
            .returning(RAGRequestLog.id)
        )
        result = await self._session.execute(stmt)
        updated_id = result.scalar_one_or_none()
        if updated_id is None:
            return None
        await self._session.flush()
        self._session.expire_all()
        return await self.get_request(updated_id)
