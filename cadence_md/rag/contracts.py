"""Stable external DTO contract for RAG service callers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Input DTO for RAG service methods."""

    query: str = Field(min_length=1)
    rag_request_id: str | None = None
    user_id: str | None = None
    clarification_answer: str | None = None
    allow_clarification: bool = True


class RAGSource(BaseModel):
    """Serializable source row aligned with retrieval/rerank ordering."""

    rank: int = Field(ge=1)
    doc_ref: str
    filename: str | None = None
    source_path: str | None = None
    document_title: str | None = None
    section_title: str | None = None
    section_id: str | None = None
    chunk_id: str | None = None
    content: str | None = None
    score: float | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None


class RAGFlags(BaseModel):
    """Operational flags describing pipeline fallbacks and truncation."""

    rerank_fallback: bool = False
    retrieval_failed: bool = False
    generate_fallback: bool = False
    context_truncated: bool = False
    query_rewritten: bool = False
    query_rewrite_fallback: bool = False
    requires_clarification: bool = False
    context_relevance_failed: bool = False
    context_relevance_fallback: bool = False
    max_query_rewrite_iterations_reached: bool = False
    answer_formatted: bool = False
    answer_format_fallback: bool = False


class RAGLatency(BaseModel):
    """Per-stage latency telemetry in milliseconds."""

    query_rewrite: float | None = None
    qdrant: float | None = None
    rerank: float | None = None
    context_relevance: float | None = None
    llm: float | None = None
    answer_format: float | None = None
    total_ms: float | None = None


class RAGResponse(BaseModel):
    """Stable full response DTO for API/worker callers."""

    query: str
    answer: str
    query_hash: str
    sources: list[RAGSource] = Field(default_factory=list)
    flags: RAGFlags = Field(default_factory=RAGFlags)
    latency: RAGLatency = Field(default_factory=RAGLatency)
    context_chars: int = 0
    answer_word_count: int = 0
    error_type: str | None = None
    error_message: str | None = None
    langfuse_trace_id: str | None = None
    retrieval_query: str | None = None
    rewritten_queries: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    raw_answer: str | None = None
    context_relevance_score: float | None = None


class RAGRetrieveResponse(BaseModel):
    """Retriever-only DTO with ranked sources and telemetry."""

    query: str
    query_hash: str
    sources: list[RAGSource] = Field(default_factory=list)
    flags: RAGFlags = Field(default_factory=RAGFlags)
    latency: RAGLatency = Field(default_factory=RAGLatency)
    error_type: str | None = None
    error_message: str | None = None
    retrieval_query: str | None = None
    rewritten_queries: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    context_relevance_score: float | None = None
