"""Chat route contracts for asynchronous RAG requests."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from cadence_md.backend.schemas.chat import CreateRAGRequest, RAGRequestStatusResponse
from cadence_md.backend.schemas.errors import ErrorResponse
from cadence_md.backend.schemas.limits import (
    CHAT_RATE_LIMIT,
    GLOBAL_QUEUE_LIMIT,
    MAX_QUERY_LENGTH,
)

router = APIRouter(prefix="/chat", tags=["chat"])

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


@router.post(
    "/messages",
    response_model=RAGRequestStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=CHAT_LIMIT_RESPONSES,
    summary="Create an asynchronous RAG request",
    description=(
        "Creates a queued RAG request and returns its id/status for polling. "
        f"Maximum query length: {MAX_QUERY_LENGTH} characters. "
        f"Per-user rate limit: {CHAT_RATE_LIMIT}. "
        f"Global queue limit: {GLOBAL_QUEUE_LIMIT}."
    ),
)
async def create_rag_message(_: CreateRAGRequest) -> RAGRequestStatusResponse:
    """Create a queued RAG request for worker processing."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="RAG request creation is not implemented yet.",
    )


@router.get(
    "/messages/{request_id}",
    response_model=RAGRequestStatusResponse,
    responses={404: {"model": ErrorResponse, "description": "RAG request was not found."}},
    summary="Poll RAG request status",
    description=(
        "Returns request status. The answer and sources are present only when status is succeeded."
    ),
)
async def get_rag_message(request_id: str) -> RAGRequestStatusResponse:
    """Poll the current status and result of a RAG request."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"RAG request polling is not implemented yet: {request_id}.",
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
async def cancel_rag_message(request_id: str) -> RAGRequestStatusResponse:
    """Cancel a queued or running RAG request if possible."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"RAG request cancellation is not implemented yet: {request_id}.",
    )


@router.post(
    "/messages/{request_id}/retry",
    response_model=RAGRequestStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=CHAT_LIMIT_RESPONSES
    | {404: {"model": ErrorResponse, "description": "Original RAG request was not found."}},
    summary="Retry a RAG request",
    description=(
        "Creates a new retry run or request with original_request_id "
        "pointing to the source request."
    ),
)
async def retry_rag_message(request_id: str) -> RAGRequestStatusResponse:
    """Create a retry request linked to the original RAG request."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"RAG request retry is not implemented yet: {request_id}.",
    )
