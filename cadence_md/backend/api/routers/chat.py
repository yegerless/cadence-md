"""Chat routes for asynchronous RAG requests."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from cadence_md.backend.api.deps import get_backend_settings, get_current_user, get_rag_enqueue
from cadence_md.backend.chat_rate_limit import chat_rate_limit_key
from cadence_md.backend.exceptions import ApiError
from cadence_md.backend.limiter import limiter
from cadence_md.backend.mappers.rag_status import rag_request_to_status_response
from cadence_md.backend.query_hash import compute_query_hash
from cadence_md.backend.request_context import get_route_settings
from cadence_md.backend.schemas.chat import CreateRAGRequest, RAGRequestStatusResponse
from cadence_md.backend.schemas.errors import ErrorResponse
from cadence_md.backend.schemas.limits import (
    CHAT_RATE_LIMIT,
    GLOBAL_QUEUE_LIMIT,
    QUEUE_LIMIT_ERROR_CODE,
)
from cadence_md.backend.services.rag_enqueue import RAGEnqueueService, make_rag_task_id
from cadence_md.backend.settings import BackendSettings
from cadence_md.db.enums import RAGRequestStatus as DbRAGRequestStatus
from cadence_md.db.models import User
from cadence_md.db.repositories.rag_logs import RAGLogRepository
from cadence_md.db.session import get_async_session

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

CHAT_LIMIT_RESPONSES = {
    429: {
        "model": ErrorResponse,
        "description": f"Rate limit exceeded: {CHAT_RATE_LIMIT}.",
    },
    503: {
        "model": ErrorResponse,
        "description": f"Global queue limit exceeded: {GLOBAL_QUEUE_LIMIT}.",
    },
}


def _chat_user_limit() -> str:
    """SlowAPI dynamic limit string from request-scoped settings."""
    return get_route_settings().CHAT_RATE_LIMIT_USER


def _effective_idempotency_key(
    body: CreateRAGRequest,
    idempotency_header: str | None,
) -> str | None:
    header_key = idempotency_header.strip() if idempotency_header else None
    body_key = body.idempotency_key.strip() if body.idempotency_key else None
    if header_key and body_key and header_key != body_key:
        raise ApiError(
            status_code=422,
            code="idempotency_key_mismatch",
            message="Idempotency-Key header and body idempotency_key must match when both are set.",
        )
    return header_key or body_key


async def _ensure_queue_capacity(session: AsyncSession, settings: BackendSettings) -> None:
    repo = RAGLogRepository(session)
    if await repo.count_active_requests() >= settings.GLOBAL_RAG_QUEUE_MAX:
        raise ApiError(
            status_code=503,
            code=QUEUE_LIMIT_ERROR_CODE,
            message="System queue capacity exceeded. Try again later.",
        )


async def _enqueue_or_fail(
    *,
    repo: RAGLogRepository,
    session: AsyncSession,
    enqueue: RAGEnqueueService,
    request_id: uuid.UUID,
) -> None:
    """Send a persisted request to the queue or mark it failed if dispatch fails."""
    try:
        await enqueue.enqueue(request_id)
    except Exception as exc:
        logger.exception(
            "Failed to enqueue RAG request",
            extra={"rag_request_id": str(request_id), "error_type": type(exc).__name__},
        )
        await repo.mark_failed(request_id)
        await session.commit()
        raise ApiError(
            status_code=503,
            code="rag_enqueue_failed",
            message="Could not enqueue RAG request. Try again later.",
        ) from exc


@router.post(
    "/messages",
    response_model=RAGRequestStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=CHAT_LIMIT_RESPONSES,
    summary="Create an asynchronous RAG request",
    description=(
        "Creates a queued RAG request and returns its id/status for polling. "
        "Clients may send Idempotency-Key header or body idempotency_key "
        "(header wins when both match). "
        f"Maximum query length: see schema. Rate limit: {CHAT_RATE_LIMIT}. "
        f"Global queue: {GLOBAL_QUEUE_LIMIT}."
    ),
)
@limiter.limit(_chat_user_limit, key_func=chat_rate_limit_key)
async def create_rag_message(
    request: Request,
    body: CreateRAGRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    settings: BackendSettings = Depends(get_backend_settings),
    enqueue: RAGEnqueueService = Depends(get_rag_enqueue),
    idempotency_header: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> RAGRequestStatusResponse:
    """Create a queued RAG request for worker processing."""
    effective_key = _effective_idempotency_key(body, idempotency_header)
    repo = RAGLogRepository(session)

    if effective_key:
        existing = await repo.get_request_by_idempotency_key(
            user_id=user.id,
            idempotency_key=effective_key,
        )
        if existing is not None:
            return await rag_request_to_status_response(repo, existing)

    await _ensure_queue_capacity(session, settings)

    qh = compute_query_hash(body.query)
    try:
        request_log = await repo.create_request(
            user_id=user.id,
            query=body.query,
            query_hash=qh,
            conversation_id=body.conversation_id,
            idempotency_key=effective_key,
        )
        request_log.celery_task_id = make_rag_task_id(request_log.id)
        await session.flush()
        await session.commit()
    except IntegrityError:
        await session.rollback()
        if effective_key:
            duplicate = await repo.get_request_by_idempotency_key(
                user_id=user.id,
                idempotency_key=effective_key,
            )
            if duplicate is not None:
                return await rag_request_to_status_response(repo, duplicate)
        raise ApiError(
            status_code=409,
            code="idempotency_conflict",
            message="Could not create request due to a conflicting idempotency key.",
        ) from None

    await _enqueue_or_fail(repo=repo, session=session, enqueue=enqueue, request_id=request_log.id)

    return await rag_request_to_status_response(repo, request_log)


@router.get(
    "/messages/{request_id}",
    response_model=RAGRequestStatusResponse,
    responses={404: {"model": ErrorResponse, "description": "RAG request was not found."}},
    summary="Poll RAG request status",
    description=(
        "Returns request status. The answer and sources are present only when status is succeeded."
    ),
)
async def get_rag_message(
    request_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> RAGRequestStatusResponse:
    """Poll the current status and result of a RAG request."""
    repo = RAGLogRepository(session)
    row = await repo.get_request_for_user(request_id=request_id, user_id=user.id)
    if row is None:
        raise ApiError(
            status_code=404,
            code="not_found",
            message="RAG request not found.",
        )
    return await rag_request_to_status_response(repo, row)


@router.post(
    "/messages/{request_id}/cancel",
    response_model=RAGRequestStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "RAG request was not found."},
        409: {"model": ErrorResponse, "description": "RAG request can no longer be cancelled."},
    },
    summary="Cancel a RAG request",
    description="Cancels a queued or running request when cancellation is still possible.",
)
async def cancel_rag_message(
    request_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> RAGRequestStatusResponse:
    """Cancel a queued or running RAG request if possible."""
    repo = RAGLogRepository(session)
    row = await repo.get_request_for_user(request_id=request_id, user_id=user.id)
    if row is None:
        raise ApiError(
            status_code=404,
            code="not_found",
            message="RAG request not found.",
        )

    if row.status == DbRAGRequestStatus.QUEUED:
        updated = await repo.mark_cancelled(row.id)
    elif row.status == DbRAGRequestStatus.RUNNING:
        updated = await repo.request_cancel(row.id)
    else:
        raise ApiError(
            status_code=409,
            code="cancellation_not_allowed",
            message="This request can no longer be cancelled.",
        )

    assert updated is not None
    return await rag_request_to_status_response(repo, updated)


@router.post(
    "/messages/{request_id}/retry",
    response_model=RAGRequestStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=CHAT_LIMIT_RESPONSES
    | {
        404: {"model": ErrorResponse, "description": "Original RAG request was not found."},
        409: {"model": ErrorResponse, "description": "Retry is not allowed for this request."},
    },
    summary="Retry a RAG request",
    description=(
        "Creates a new retry run or request with original_request_id "
        "pointing to the source request."
    ),
)
@limiter.limit(_chat_user_limit, key_func=chat_rate_limit_key)
async def retry_rag_message(
    request: Request,
    request_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    settings: BackendSettings = Depends(get_backend_settings),
    enqueue: RAGEnqueueService = Depends(get_rag_enqueue),
) -> RAGRequestStatusResponse:
    """Create a retry request linked to the original RAG request."""
    repo = RAGLogRepository(session)
    row = await repo.get_request_for_user(request_id=request_id, user_id=user.id)
    if row is None:
        raise ApiError(
            status_code=404,
            code="not_found",
            message="RAG request not found.",
        )

    if row.status in (DbRAGRequestStatus.QUEUED, DbRAGRequestStatus.RUNNING):
        raise ApiError(
            status_code=409,
            code="retry_not_allowed",
            message="Retry is only allowed after the request has finished processing.",
        )

    await _ensure_queue_capacity(session, settings)

    try:
        new_request = await repo.create_retry_request(original_request_id=row.id)
        new_request.celery_task_id = make_rag_task_id(new_request.id)
        await session.flush()
        await session.commit()
        await _enqueue_or_fail(
            repo=repo,
            session=session,
            enqueue=enqueue,
            request_id=new_request.id,
        )
    except ValueError as exc:
        raise ApiError(
            status_code=404,
            code="not_found",
            message="Original RAG request was not found.",
        ) from exc

    return await rag_request_to_status_response(repo, new_request)
