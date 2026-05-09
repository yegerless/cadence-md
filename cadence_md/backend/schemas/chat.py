"""Chat and asynchronous RAG request schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from cadence_md.backend.schemas.limits import MAX_QUERY_LENGTH


class RAGRequestStatus(StrEnum):
    """Lifecycle statuses for an asynchronous RAG request."""

    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CreateChatConversationRequest(BaseModel):
    """Payload for creating an empty chat conversation."""

    title: str | None = Field(
        default=None,
        max_length=255,
        description="Optional user-visible chat title.",
    )


class ChatConversationResponse(BaseModel):
    """User-owned chat conversation metadata."""

    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None


class ChatConversationListResponse(BaseModel):
    """Paginated chat conversation list."""

    items: list[ChatConversationResponse]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class CreateRAGRequest(BaseModel):
    """Payload for creating an asynchronous RAG request."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="Doctor's clinical question for the RAG assistant.",
        examples=["Какие препараты первой линии рекомендованы при артериальной гипертензии?"],
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional frontend conversation id for grouping requests.",
    )
    chat_id: str | None = Field(
        default=None,
        description="Optional persisted chat conversation id. A new chat is created when omitted.",
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=255,
        description="Optional idempotency token when the Idempotency-Key header is not used.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional lightweight client metadata for tracing.",
    )


class RAGSourceResponse(BaseModel):
    """Source document excerpt returned with a successful RAG answer."""

    rank: int = Field(ge=1)
    doc_ref: str = Field(description="Citation marker used in the answer, e.g. [Doc 1].")
    filename: str | None = None
    source_path: str | None = None
    document_title: str | None = None
    section_title: str | None = None
    section_id: str | None = None
    chunk_id: str | None = None
    content: str | None = Field(default=None, description="Retrieved chunk text.")
    score: float | None = None


class RAGAnswerResponse(BaseModel):
    """Completed RAG answer payload."""

    answer: str
    sources: list[RAGSourceResponse] = Field(default_factory=list)
    query_hash: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    flags: dict[str, bool] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    langfuse_trace_id: str | None = None


class ClarificationResponse(BaseModel):
    """Pending clarification prompt returned while a request waits for the user."""

    question: str
    answered: bool = False
    requested_at: datetime | None = None


class SubmitClarificationRequest(BaseModel):
    """Payload for resuming a RAG request after user clarification."""

    answer: str = Field(
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="User clarification answer used to continue the existing RAG request.",
        examples=["Речь о взрослых пациентах с впервые выявленной гипертензией без ХБП."],
    )


class RAGRequestStatusResponse(BaseModel):
    """Polling response for an asynchronous RAG request."""

    request_id: str
    status: RAGRequestStatus
    chat_id: str | None = Field(
        default=None,
        description="Persisted chat conversation id when the request belongs to a chat.",
    )
    original_request_id: str | None = Field(
        default=None,
        description="Original request id when this status belongs to a retry request.",
    )
    answer: RAGAnswerResponse | None = Field(
        default=None,
        description="Present only when status is succeeded.",
    )
    error: str | None = Field(
        default=None,
        description="Failure summary when status is failed.",
    )
    clarification: ClarificationResponse | None = Field(
        default=None,
        description="Present only while the request is awaiting user clarification.",
    )


class ChatTurnResponse(BaseModel):
    """Single chat turn reconstructed from a persisted RAG request."""

    query: str
    request: RAGRequestStatusResponse


class ChatMessageHistoryResponse(BaseModel):
    """Chronological chat message history."""

    items: list[ChatTurnResponse]
