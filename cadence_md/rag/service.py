"""Service adapter that maps internal RAGState to stable DTO contracts."""

from __future__ import annotations

from typing import Any

from cadence_md.app.rag import RAGPipeline
from cadence_md.rag.contracts import (
    RAGFlags,
    RAGLatency,
    RAGRequest,
    RAGResponse,
    RAGRetrieveResponse,
    RAGSource,
)


class RAGService:
    """Stable entrypoint for callers that must not depend on RAGState internals."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def run(self, request: RAGRequest) -> RAGResponse:
        state = self.pipeline.run(request.query)
        return self._state_to_response(state)

    def retrieve(self, request: RAGRequest) -> RAGRetrieveResponse:
        state = self.pipeline.build_initial_state(request.query)
        state = self.pipeline.retrieve_node(state)
        state = self.pipeline.reranker_node(state)
        return self._state_to_retrieve_response(state)

    def _state_to_sources(self, state: dict[str, Any]) -> list[RAGSource]:
        ranked_docs = state.get("ranked_docs") or []
        sources: list[RAGSource] = []
        for idx, ranked in enumerate(ranked_docs, start=1):
            doc = ranked["doc"]
            meta = doc.metadata or {}
            sources.append(
                RAGSource(
                    rank=idx,
                    doc_ref=f"[Doc {idx}]",
                    filename=meta.get("filename"),
                    document_title=meta.get("document_title"),
                    section_title=meta.get("section_title"),
                    section_id=meta.get("section_id"),
                    chunk_id=meta.get("chunk_id"),
                    content=doc.page_content,
                    score=ranked.get("final_score"),
                    retrieval_score=ranked.get("retrieval_score"),
                    rerank_score=ranked.get("rerank_score"),
                )
            )
        return sources

    def _state_to_flags(self, state: dict[str, Any]) -> RAGFlags:
        return RAGFlags(
            rerank_fallback=bool(state.get("rerank_fallback", False)),
            retrieval_failed=bool(state.get("retrieval_failed", False)),
            generate_fallback=bool(state.get("generate_fallback", False)),
            context_truncated=bool(state.get("context_truncated", False)),
        )

    def _state_to_latency(self, state: dict[str, Any]) -> RAGLatency:
        latency: dict[str, float] = state.get("latency_ms") or {}
        total_ms = sum(latency.values()) if latency else None
        return RAGLatency(
            qdrant=latency.get("qdrant"),
            rerank=latency.get("rerank"),
            llm=latency.get("llm"),
            total_ms=total_ms,
        )

    def _state_to_response(self, state: dict[str, Any]) -> RAGResponse:
        return RAGResponse(
            query=str(state.get("query", "")),
            answer=str(state.get("answer", "")),
            query_hash=str(state.get("query_hash", "")),
            sources=self._state_to_sources(state),
            flags=self._state_to_flags(state),
            latency=self._state_to_latency(state),
            context_chars=int(state.get("context_chars", 0)),
            answer_word_count=int(state.get("answer_word_count", 0)),
            error_type=state.get("error_type"),
            error_message=state.get("error_message"),
        )

    def _state_to_retrieve_response(self, state: dict[str, Any]) -> RAGRetrieveResponse:
        return RAGRetrieveResponse(
            query=str(state.get("query", "")),
            query_hash=str(state.get("query_hash", "")),
            sources=self._state_to_sources(state),
            flags=self._state_to_flags(state),
            latency=self._state_to_latency(state),
            error_type=state.get("error_type"),
            error_message=state.get("error_message"),
        )
