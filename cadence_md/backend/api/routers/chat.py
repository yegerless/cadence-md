"""Chat routes for asynchronous RAG requests."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from cadence_md.backend.api.deps import get_backend_settings, get_current_user, get_rag_enqueue
from cadence_md.backend.chat_rate_limit import chat_rate_limit_key
from cadence_md.backend.exceptions import ApiError
from cadence_md.backend.limiter import limiter
from cadence_md.backend.mappers.rag_status import rag_request_to_status_response
from cadence_md.backend.query_hash import compute_query_hash
from cadence_md.backend.request_context import get_route_settings
from cadence_md.backend.schemas.chat import (
    ChatConversationListResponse,
    ChatConversationResponse,
    ChatMessageHistoryResponse,
    ChatTurnResponse,
    CreateChatConversationRequest,
    CreateRAGRequest,
    RAGRequestStatusResponse,
    SubmitClarificationRequest,
)
from cadence_md.backend.schemas.errors import ErrorResponse
from cadence_md.backend.schemas.limits import (
    CHAT_RATE_LIMIT,
    GLOBAL_QUEUE_LIMIT,
    QUEUE_LIMIT_ERROR_CODE,
)
from cadence_md.backend.services.rag_enqueue import RAGEnqueueService, make_rag_task_id
from cadence_md.backend.settings import BackendSettings
from cadence_md.db.enums import RAGRequestStatus as DbRAGRequestStatus
from cadence_md.db.models import ChatConversation, User
from cadence_md.db.repositories.chat_conversations import ChatConversationRepository
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


def _chat_to_response(chat: ChatConversation) -> ChatConversationResponse:
    """Map chat metadata to the public API shape."""
    return ChatConversationResponse(
        id=str(chat.id),
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        last_message_at=chat.last_message_at,
    )


async def _get_active_chat_or_404(
    repo: ChatConversationRepository,
    *,
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
) -> ChatConversation:
    """Return an active user-owned chat or hide its existence."""
    chat = await repo.get_active_chat_for_user(user_id=user_id, chat_id=chat_id)
    if chat is None:
        raise ApiError(
            status_code=404,
            code="not_found",
            message="Chat conversation not found.",
        )
    return chat


def _parse_chat_id_or_404(chat_id: str) -> uuid.UUID:
    """Parse a client-provided chat id without exposing malformed identifiers."""
    try:
        return uuid.UUID(chat_id)
    except ValueError as exc:
        raise ApiError(
            status_code=404,
            code="not_found",
            message="Chat conversation not found.",
        ) from exc


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


async def _enqueue_clarification_or_503(
    *,
    enqueue: RAGEnqueueService,
    request_id: uuid.UUID,
    task_id: str,
) -> None:
    """Send a clarification resume task without changing persisted request state on failure."""
    try:
        await enqueue.enqueue(request_id, task_id=task_id)
    except Exception as exc:
        logger.exception(
            "Failed to enqueue clarified RAG request",
            extra={"rag_request_id": str(request_id), "error_type": type(exc).__name__},
        )
        raise ApiError(
            status_code=503,
            code="rag_enqueue_failed",
            message="Could not enqueue RAG request. Try again later.",
        ) from exc


def _source_rank(source: dict[str, object]) -> int | None:
    """Return a source rank from persisted JSON, accepting int-like values."""
    try:
        return int(source.get("rank", 0))
    except (TypeError, ValueError):
        return None


def _find_source_by_rank(
    sources: list[dict[str, object]],
    rank: int,
) -> dict[str, object] | None:
    """Find a persisted source row by 1-based rank."""
    return next((source for source in sources if _source_rank(source) == rank), None)


def _resolve_source_file(corpus_dir: Path, source_path: str) -> Path:
    """Resolve a source path under corpus root and reject traversal or missing files."""
    corpus_root = corpus_dir.resolve()
    candidate = (corpus_root / source_path).resolve()
    try:
        candidate.relative_to(corpus_root)
    except ValueError as exc:
        raise ApiError(
            status_code=404,
            code="source_not_found",
            message="Source file was not found.",
        ) from exc
    if not candidate.is_file():
        raise ApiError(
            status_code=404,
            code="source_not_found",
            message="Source file was not found.",
        )
    return candidate


def _download_filename(source: dict[str, object], source_file: Path) -> str:
    """Return a safe download filename from source metadata or resolved file path."""
    raw_filename = source.get("filename")
    if isinstance(raw_filename, str) and raw_filename.strip():
        return Path(raw_filename).name
    return source_file.name


@router.post(
    "/conversations",
    response_model=ChatConversationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": ErrorResponse, "description": "Chat conversation was not found."}},
    summary="Create a chat conversation",
    description="Creates an empty persisted chat conversation for the current user.",
)
async def create_chat_conversation(
    body: CreateChatConversationRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> ChatConversationResponse:
    """Create an empty user-owned chat conversation."""
    repo = ChatConversationRepository(session)
    chat = await repo.create_chat(user_id=user.id, title=body.title)
    await session.commit()
    return _chat_to_response(chat)


@router.get(
    "/conversations",
    response_model=ChatConversationListResponse,
    summary="List chat conversations",
    description="Returns active chat conversations owned by the current user.",
)
async def list_chat_conversations(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChatConversationListResponse:
    """List active chats for the current user."""
    repo = ChatConversationRepository(session)
    chats = await repo.list_active_chats_for_user(user_id=user.id, limit=limit, offset=offset)
    return ChatConversationListResponse(
        items=[_chat_to_response(chat) for chat in chats],
        limit=limit,
        offset=offset,
    )


@router.get(
    "/conversations/{chat_id}",
    response_model=ChatConversationResponse,
    responses={404: {"model": ErrorResponse, "description": "Chat conversation was not found."}},
    summary="Get a chat conversation",
    description="Returns active chat metadata when it belongs to the current user.",
)
async def get_chat_conversation(
    chat_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> ChatConversationResponse:
    """Return a single active user-owned chat conversation."""
    repo = ChatConversationRepository(session)
    chat = await _get_active_chat_or_404(repo, user_id=user.id, chat_id=chat_id)
    return _chat_to_response(chat)


@router.get(
    "/conversations/{chat_id}/messages",
    response_model=ChatMessageHistoryResponse,
    responses={404: {"model": ErrorResponse, "description": "Chat conversation was not found."}},
    summary="Get chat conversation messages",
    description="Returns chronological RAG turns for an active user-owned chat.",
)
async def get_chat_conversation_messages(
    chat_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> ChatMessageHistoryResponse:
    """Return persisted turns for an active chat in chronological order."""
    chat_repo = ChatConversationRepository(session)
    await _get_active_chat_or_404(chat_repo, user_id=user.id, chat_id=chat_id)
    rag_repo = RAGLogRepository(session)
    requests = await chat_repo.list_requests_by_chat_for_user(user_id=user.id, chat_id=chat_id)
    return ChatMessageHistoryResponse(
        items=[
            ChatTurnResponse(
                query=request_log.query,
                request=await rag_request_to_status_response(rag_repo, request_log),
            )
            for request_log in requests
        ]
    )


@router.delete(
    "/conversations/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse, "description": "Chat conversation was not found."}},
    summary="Soft delete a chat conversation",
    description="Marks a chat conversation as deleted without deleting its RAG requests.",
)
async def delete_chat_conversation(
    chat_id: uuid.UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Soft-delete a user-owned chat conversation."""
    repo = ChatConversationRepository(session)
    chat = await repo.soft_delete_chat_for_user(user_id=user.id, chat_id=chat_id)
    if chat is None:
        raise ApiError(
            status_code=404,
            code="not_found",
            message="Chat conversation not found.",
        )
    await session.commit()


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
    chat_repo = ChatConversationRepository(session)

    if effective_key:
        existing = await repo.get_request_by_idempotency_key(
            user_id=user.id,
            idempotency_key=effective_key,
        )
        if existing is not None:
            return await rag_request_to_status_response(repo, existing)

    chat_id = None
    if body.chat_id is not None:
        parsed_chat_id = _parse_chat_id_or_404(body.chat_id)
        chat = await _get_active_chat_or_404(chat_repo, user_id=user.id, chat_id=parsed_chat_id)
        chat_id = chat.id

    await _ensure_queue_capacity(session, settings)

    qh = compute_query_hash(body.query)
    try:
        if chat_id is None:
            chat = await chat_repo.create_chat(user_id=user.id)
            chat_id = chat.id

        request_log = await repo.create_request(
            user_id=user.id,
            query=body.query,
            query_hash=qh,
            chat_id=chat_id,
            conversation_id=body.conversation_id,
            idempotency_key=effective_key,
        )
        request_log.celery_task_id = make_rag_task_id(request_log.id)
        await chat_repo.touch_chat_for_user(
            user_id=user.id,
            chat_id=chat_id,
            last_message_at=request_log.created_at,
        )
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
    "/messages/{request_id}/clarification",
    response_model=RAGRequestStatusResponse,
    responses=CHAT_LIMIT_RESPONSES
    | {
        404: {"model": ErrorResponse, "description": "RAG request was not found."},
        409: {"model": ErrorResponse, "description": "Clarification is not allowed."},
    },
    summary="Submit clarification for a RAG request",
    description="Stores a user clarification answer and requeues the existing RAG request.",
)
@limiter.limit(_chat_user_limit, key_func=chat_rate_limit_key)
async def submit_rag_clarification(
    request: Request,
    request_id: uuid.UUID,
    body: SubmitClarificationRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    enqueue: RAGEnqueueService = Depends(get_rag_enqueue),
) -> RAGRequestStatusResponse:
    """Resume a RAG request that is waiting for user clarification."""
    repo = RAGLogRepository(session)
    row = await repo.get_request_for_user(request_id=request_id, user_id=user.id)
    if row is None:
        raise ApiError(
            status_code=404,
            code="not_found",
            message="RAG request not found.",
        )

    if row.status == DbRAGRequestStatus.QUEUED and row.clarification_answer:
        return await rag_request_to_status_response(repo, row)
    if row.status != DbRAGRequestStatus.AWAITING_CLARIFICATION:
        raise ApiError(
            status_code=409,
            code="clarification_not_allowed",
            message="Clarification is only accepted while the request is awaiting clarification.",
        )

    task_id = make_rag_task_id(row.id, clarification_attempt=row.clarification_attempts)
    updated = await repo.submit_clarification(
        row.id,
        answer=body.answer,
        celery_task_id=task_id,
    )
    if updated is None:
        raise ApiError(
            status_code=409,
            code="clarification_not_allowed",
            message="Clarification is only accepted while the request is awaiting clarification.",
        )
    await session.commit()

    await _enqueue_clarification_or_503(enqueue=enqueue, request_id=updated.id, task_id=task_id)
    return await rag_request_to_status_response(repo, updated)


@router.get(
    "/messages/{request_id}/sources/{rank}/download",
    response_class=FileResponse,
    responses={
        404: {"model": ErrorResponse, "description": "RAG request or source file was not found."},
        409: {"model": ErrorResponse, "description": "RAG request has no completed sources yet."},
    },
    summary="Download a RAG source PDF",
    description="Downloads one source PDF from a completed RAG answer by source rank.",
)
async def download_rag_source(
    request_id: uuid.UUID,
    rank: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    settings: BackendSettings = Depends(get_backend_settings),
) -> FileResponse:
    """Download a source file that belongs to the current user's completed RAG request."""
    repo = RAGLogRepository(session)
    row = await repo.get_request_for_user(request_id=request_id, user_id=user.id)
    if row is None:
        raise ApiError(
            status_code=404,
            code="not_found",
            message="RAG request not found.",
        )
    if row.status != DbRAGRequestStatus.SUCCEEDED:
        raise ApiError(
            status_code=409,
            code="source_download_not_available",
            message="Source downloads are available only for completed RAG requests.",
        )

    response_row = await repo.get_response_for_request(row.id)
    if response_row is None:
        raise ApiError(
            status_code=404,
            code="source_not_found",
            message="Source file was not found.",
        )

    sources: list[dict[str, object]] = list(response_row.sources_json or [])
    source = _find_source_by_rank(sources, rank)
    source_path = source.get("source_path") if source is not None else None
    if source is None or not isinstance(source_path, str) or not source_path.strip():
        raise ApiError(
            status_code=404,
            code="source_not_found",
            message="Source file was not found.",
        )

    source_file = _resolve_source_file(settings.RAG_CORPUS_DIR, source_path)
    return FileResponse(
        path=source_file,
        media_type="application/pdf",
        filename=_download_filename(source, source_file),
    )


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

    if row.status in (
        DbRAGRequestStatus.QUEUED,
        DbRAGRequestStatus.AWAITING_CLARIFICATION,
    ):
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

    if row.status in (
        DbRAGRequestStatus.QUEUED,
        DbRAGRequestStatus.RUNNING,
        DbRAGRequestStatus.AWAITING_CLARIFICATION,
    ):
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
