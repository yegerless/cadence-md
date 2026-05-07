"""Map ``RAGRequestLog`` / ``RAGResponseLog`` rows to polling responses."""

from __future__ import annotations

from cadence_md.backend.schemas.chat import (
    ClarificationResponse,
    RAGAnswerResponse,
    RAGRequestStatusResponse,
    RAGSourceResponse,
)
from cadence_md.backend.schemas.chat import (
    RAGRequestStatus as ApiRAGRequestStatus,
)
from cadence_md.db.enums import RAGRequestStatus as DbRAGRequestStatus
from cadence_md.db.models import RAGRequestLog
from cadence_md.db.repositories.rag_logs import RAGLogRepository


async def rag_request_to_status_response(
    repo: RAGLogRepository,
    request_log: RAGRequestLog,
) -> RAGRequestStatusResponse:
    """Build the API polling payload for a single request row."""
    response_row = await repo.get_response_for_request(request_log.id)

    answer = None
    error = None
    clarification = None
    if request_log.status == DbRAGRequestStatus.SUCCEEDED and response_row is not None:
        sources_raw: list[dict[str, object]] = list(response_row.sources_json or [])
        sources = [RAGSourceResponse.model_validate(item) for item in sources_raw]
        answer = RAGAnswerResponse(
            answer=response_row.answer,
            sources=sources,
            query_hash=request_log.query_hash,
            latency_ms=dict(response_row.latency_ms_json or {}),
            flags=dict(response_row.flags_json or {}),
            error_type=response_row.error_type,
            error_message=response_row.error_message,
            langfuse_trace_id=response_row.langfuse_trace_id,
        )
    elif request_log.status == DbRAGRequestStatus.FAILED:
        error = (
            response_row.error_message
            if response_row is not None and response_row.error_message
            else "Request failed."
        )
    elif (
        request_log.status == DbRAGRequestStatus.AWAITING_CLARIFICATION
        and request_log.clarification_question
    ):
        clarification = ClarificationResponse(
            question=request_log.clarification_question,
            answered=False,
            requested_at=request_log.clarification_requested_at,
        )

    original_request_id = (
        str(request_log.original_request_id) if request_log.original_request_id else None
    )

    return RAGRequestStatusResponse(
        request_id=str(request_log.id),
        status=ApiRAGRequestStatus(request_log.status.value),
        original_request_id=original_request_id,
        answer=answer,
        error=error,
        clarification=clarification,
    )
