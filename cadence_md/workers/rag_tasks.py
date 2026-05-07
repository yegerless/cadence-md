"""Celery tasks for asynchronous RAG request processing."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import worker_process_init, worker_process_shutdown, worker_ready
from kombu.exceptions import KombuError
from redis.exceptions import RedisError
from sqlalchemy.exc import DBAPIError, OperationalError

from cadence_md.backend.settings import backend_settings
from cadence_md.db.enums import RAGRequestStatus
from cadence_md.db.models import RAGRequestLog
from cadence_md.db.repositories import DuplicateRAGResponseError, RAGLogRepository
from cadence_md.db.session import AsyncSessionLocal
from cadence_md.observability.logging import log_context
from cadence_md.observability.metrics import record_celery_task_result, update_queue_depth
from cadence_md.rag.contracts import RAGRequest, RAGResponse
from cadence_md.workers.celery_app import celery_app
from cadence_md.workers.metrics_server import start_worker_metrics_server
from cadence_md.workers.rag_bootstrap import get_rag_runtime, shutdown_rag_runtime

logger = logging.getLogger(__name__)

INFRA_TRANSIENT_ERRORS = (OperationalError, DBAPIError, RedisError, KombuError)
TERMINAL_STATUSES = {
    RAGRequestStatus.SUCCEEDED,
    RAGRequestStatus.FAILED,
    RAGRequestStatus.CANCELLED,
}


@worker_process_init.connect
def _warmup_worker_process(**_: object) -> None:
    """Build the RAG runtime once in each Celery worker process."""
    get_rag_runtime()


@worker_process_shutdown.connect
def _shutdown_worker_process(**_: object) -> None:
    """Close RAG runtime resources when Celery shuts down the worker process."""
    shutdown_rag_runtime()
    asyncio.run(AsyncSessionLocal.kw["bind"].dispose())


@worker_ready.connect
def _start_worker_metrics_endpoint(**_: object) -> None:
    """Expose worker Prometheus metrics after the Celery worker is ready."""
    start_worker_metrics_server()


@celery_app.task(
    bind=True,
    name=backend_settings.CELERY_RAG_TASK_NAME,
    autoretry_for=INFRA_TRANSIENT_ERRORS,
    retry_backoff=backend_settings.CELERY_TASK_RETRY_BACKOFF_SECONDS,
    retry_jitter=True,
    retry_kwargs={"max_retries": backend_settings.CELERY_TASK_MAX_RETRIES},
    soft_time_limit=backend_settings.CELERY_TASK_SOFT_TIME_LIMIT_SECONDS,
    time_limit=backend_settings.CELERY_TASK_TIME_LIMIT_SECONDS,
)
def run_rag_request(self: Task, rag_request_id: str) -> str:
    """Celery entrypoint for running a persisted RAG request."""
    celery_task_id = str(self.request.id) if self.request.id else None
    with log_context(rag_request_id=rag_request_id, task_id=celery_task_id):
        return _run_rag_request_in_fresh_event_loop(rag_request_id, celery_task_id=celery_task_id)


def _run_rag_request_in_fresh_event_loop(
    rag_request_id: str,
    *,
    celery_task_id: str | None,
) -> str:
    """Run the async RAG task body in a new loop and drop the DB pool for that loop.

    Celery calls this once per task; each ``asyncio.run`` creates a new event loop.
    Asyncpg connections must not be reused across loops, so we dispose the pool bound
    to ``AsyncSessionLocal`` before the loop closes.
    """
    return asyncio.run(
        _run_rag_request_async_with_engine_cleanup(rag_request_id, celery_task_id=celery_task_id)
    )


async def _run_rag_request_async_with_engine_cleanup(
    rag_request_id: str,
    *,
    celery_task_id: str | None,
) -> str:
    """Delegate to :func:`_run_rag_request_async` and always dispose the session bind's pool."""
    try:
        return await _run_rag_request_async(rag_request_id, celery_task_id=celery_task_id)
    finally:
        await AsyncSessionLocal.kw["bind"].dispose()


async def _run_rag_request_async(
    rag_request_id: str,
    *,
    celery_task_id: str | None,
) -> str:
    """Run a RAG request and persist the terminal state."""
    started = time.perf_counter()
    result = "failed"
    request_id = uuid.UUID(rag_request_id)
    with log_context(rag_request_id=rag_request_id, task_id=celery_task_id):
        try:
            async with AsyncSessionLocal() as session:
                repo = RAGLogRepository(session)
                row = await repo.get_request(request_id)
                if row is None:
                    logger.warning(
                        "RAG task request not found",
                        extra={"rag_request_id": rag_request_id},
                    )
                    result = "not_found"
                elif await _is_already_done(repo, row):
                    result = "already_done"
                elif row.status == RAGRequestStatus.QUEUED and row.cancel_requested_at is not None:
                    await repo.mark_cancelled(row.id)
                    await session.commit()
                    result = "cancelled"
                else:
                    claimed = await repo.claim_queued_request(row.id, celery_task_id=celery_task_id)
                    if claimed is None:
                        await session.rollback()
                        result = await _handle_unclaimed_request(repo, request_id)
                    else:
                        await session.commit()
                        result = await _run_claimed_request(
                            repo,
                            session,
                            claimed,
                            request_id,
                            rag_request_id,
                        )
                return result
        finally:
            record_celery_task_result(
                backend_settings.CELERY_RAG_TASK_NAME,
                result,
                time.perf_counter() - started,
            )
            update_queue_depth(
                broker_url=backend_settings.CELERY_BROKER_URL,
                queue_name=backend_settings.CELERY_RAG_QUEUE_NAME,
            )


async def _run_claimed_request(
    repo: RAGLogRepository,
    session: Any,
    claimed: RAGRequestLog,
    request_id: uuid.UUID,
    rag_request_id: str,
) -> str:
    """Run RAG for a request that this task has successfully claimed."""
    if await _cancel_requested(repo, session, request_id):
        await repo.mark_cancelled_from_running(request_id)
        await session.commit()
        return "cancelled"

    try:
        response = get_rag_runtime().service.run(
            RAGRequest(
                query=claimed.query,
                rag_request_id=rag_request_id,
                user_id=str(claimed.user_id),
                clarification_answer=claimed.clarification_answer,
                allow_clarification=claimed.clarification_answer is None,
            )
        )
    except SoftTimeLimitExceeded as exc:
        await _persist_failure(repo, session, request_id, "soft_time_limit", str(exc))
        return "failed"
    except Exception as exc:
        logger.exception(
            "RAG task failed",
            extra={"rag_request_id": rag_request_id, "error_type": type(exc).__name__},
        )
        await _persist_failure(repo, session, request_id, type(exc).__name__, str(exc))
        return "failed"

    if await _cancel_requested(repo, session, request_id):
        await repo.mark_cancelled_from_running(request_id)
        await session.commit()
        return "cancelled"

    if (
        claimed.clarification_answer is None
        and response.flags.requires_clarification
        and response.clarification_question
    ):
        return await _persist_awaiting_clarification(
            repo,
            session,
            request_id,
            response.clarification_question,
        )

    return await _persist_success(repo, session, request_id, response)


async def _is_already_done(repo: RAGLogRepository, row: RAGRequestLog) -> bool:
    """Return whether the task should no-op for an already finalized request."""
    if row.status in TERMINAL_STATUSES:
        logger.info(
            "RAG task skipped for terminal request",
            extra={"rag_request_id": str(row.id), "status": row.status.value},
        )
        return True
    if await repo.has_response_for_request(row.id):
        logger.info(
            "RAG task skipped because response already exists",
            extra={"rag_request_id": str(row.id), "status": row.status.value},
        )
        return True
    return False


async def _handle_unclaimed_request(repo: RAGLogRepository, request_id: uuid.UUID) -> str:
    """Resolve a task that could not claim queued ownership."""
    row = await repo.get_request(request_id)
    if row is None:
        return "not_found"
    if await _is_already_done(repo, row):
        return "already_done"
    logger.info(
        "RAG task skipped because request was not claimable",
        extra={"rag_request_id": str(request_id), "status": row.status.value},
    )
    return "not_claimed"


async def _cancel_requested(
    repo: RAGLogRepository,
    session: Any,
    request_id: uuid.UUID,
) -> bool:
    """Reload a request and return whether cancellation was requested."""
    session.expire_all()
    row = await repo.get_request(request_id)
    return row is not None and row.cancel_requested_at is not None


async def _persist_success(
    repo: RAGLogRepository,
    session: Any,
    request_id: uuid.UUID,
    response: RAGResponse,
) -> str:
    """Persist successful RAG output and mark the request succeeded."""
    try:
        await repo.create_response(
            rag_request_id=request_id,
            answer=response.answer,
            sources=[source.model_dump() for source in response.sources],
            latency_ms=response.latency.model_dump(exclude_none=True),
            flags=response.flags.model_dump(),
            error_type=response.error_type,
            error_message=response.error_message,
            langfuse_trace_id=response.langfuse_trace_id,
        )
        updated = await repo.mark_succeeded_from_running(request_id)
    except DuplicateRAGResponseError:
        await session.rollback()
        return "already_done"

    if updated is None:
        await session.rollback()
        return "not_claimed"

    await session.commit()
    logger.info("RAG task succeeded", extra={"rag_request_id": str(request_id)})
    return "succeeded"


async def _persist_awaiting_clarification(
    repo: RAGLogRepository,
    session: Any,
    request_id: uuid.UUID,
    question: str,
) -> str:
    """Persist a clarification prompt without creating a final response row."""
    updated = await repo.mark_awaiting_clarification(request_id, question=question)
    if updated is None:
        await session.rollback()
        return "not_claimed"

    await session.commit()
    logger.info("RAG task awaits clarification", extra={"rag_request_id": str(request_id)})
    return "awaiting_clarification"


async def _persist_failure(
    repo: RAGLogRepository,
    session: Any,
    request_id: uuid.UUID,
    error_type: str,
    error_message: str,
) -> None:
    """Persist a failed RAG task result and terminal status."""
    try:
        await repo.create_response(
            rag_request_id=request_id,
            answer="",
            sources=[],
            latency_ms={},
            flags={},
            error_type=error_type,
            error_message=error_message,
        )
        updated = await repo.mark_failed_from_running(request_id)
    except DuplicateRAGResponseError:
        await session.rollback()
        return

    if updated is None:
        await session.rollback()
        return

    await session.commit()
