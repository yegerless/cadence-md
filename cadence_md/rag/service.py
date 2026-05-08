"""Service adapter that maps internal RAGState to stable DTO contracts."""

from __future__ import annotations

from typing import Any

from cadence_md.app.rag import RAGPipeline
from cadence_md.app.settings import settings
from cadence_md.observability.langfuse import (
    LangfuseTracer,
    active_trace,
    get_langfuse_tracer,
    record_trace_summary,
)
from cadence_md.observability.privacy import query_hash_only
from cadence_md.observability.settings import observability_settings
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

    def __init__(self, pipeline: RAGPipeline, tracer: LangfuseTracer | None = None) -> None:
        self.pipeline = pipeline
        self.tracer = tracer or get_langfuse_tracer()

    def run(self, request: RAGRequest) -> RAGResponse:
        trace = self.tracer.start_trace(
            query=request.query,
            query_hash=query_hash_only(request.query),
            rag_request_id=request.rag_request_id,
            user_id=request.user_id,
            prompt_version=settings.rag_config.prompt_version,
        )
        with active_trace(trace):
            state = self.pipeline.run(
                request.query,
                clarification_answer=request.clarification_answer,
                allow_clarification=request.allow_clarification,
            )
            response = self._state_to_response(state, langfuse_trace_id=trace.trace_id)
            record_trace_summary(**self._trace_summary(response))
        return response

    def retrieve(self, request: RAGRequest) -> RAGRetrieveResponse:
        state = self.pipeline.run_retriever_only(
            request.query,
            clarification_answer=request.clarification_answer,
            allow_clarification=request.allow_clarification,
        )
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
                    source_path=meta.get("source_path"),
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
            query_rewritten=bool(state.get("query_rewritten", False)),
            query_rewrite_fallback=bool(state.get("query_rewrite_fallback", False)),
            requires_clarification=bool(state.get("requires_clarification", False)),
            context_relevance_failed=bool(state.get("context_relevance_failed", False)),
            context_relevance_fallback=bool(state.get("context_relevance_fallback", False)),
            max_query_rewrite_iterations_reached=bool(
                state.get("max_query_rewrite_iterations_reached", False)
            ),
            answer_formatted=bool(state.get("answer_formatted", False)),
            answer_format_fallback=bool(state.get("answer_format_fallback", False)),
        )

    def _state_to_latency(self, state: dict[str, Any]) -> RAGLatency:
        latency: dict[str, float] = state.get("latency_ms") or {}
        total_ms = sum(latency.values()) if latency else None
        return RAGLatency(
            query_rewrite=latency.get("query_rewrite"),
            qdrant=latency.get("qdrant"),
            rerank=latency.get("rerank"),
            context_relevance=latency.get("context_relevance"),
            llm=latency.get("llm"),
            answer_format=latency.get("answer_format"),
            total_ms=total_ms,
        )

    def _state_to_response(
        self,
        state: dict[str, Any],
        *,
        langfuse_trace_id: str | None = None,
    ) -> RAGResponse:
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
            langfuse_trace_id=langfuse_trace_id,
            retrieval_query=state.get("retrieval_query"),
            rewritten_queries=list(state.get("rewritten_queries") or []),
            clarification_question=state.get("clarification_question"),
            raw_answer=state.get("raw_answer"),
            context_relevance_score=state.get("context_relevance_score"),
        )

    def _trace_summary(self, response: RAGResponse) -> dict[str, Any]:
        """Build redacted trace-level output and metadata for Langfuse."""
        trace_output = (
            response.answer
            if observability_settings.LANGFUSE_TRACE_QUERY_MODE == "full"
            else "[redacted rag answer]"
        )
        metadata = {
            "query_hash": response.query_hash,
            "latency_ms": response.latency.model_dump(exclude_none=True),
            "flags": response.flags.model_dump(),
            "source_count": len(response.sources),
            "sources": [
                source.model_dump(exclude={"content"}, exclude_none=True)
                for source in response.sources
            ],
            "context_chars": response.context_chars,
            "answer_word_count": response.answer_word_count,
            "retrieval_query": response.retrieval_query,
            "rewritten_queries": response.rewritten_queries,
            "context_relevance_score": response.context_relevance_score,
            "error_type": response.error_type,
            "error_message": response.error_message,
            "requires_clarification": response.flags.requires_clarification,
        }
        return {
            "output_data": trace_output,
            "metadata": {key: value for key, value in metadata.items() if value is not None},
        }

    def _state_to_retrieve_response(self, state: dict[str, Any]) -> RAGRetrieveResponse:
        return RAGRetrieveResponse(
            query=str(state.get("query", "")),
            query_hash=str(state.get("query_hash", "")),
            sources=self._state_to_sources(state),
            flags=self._state_to_flags(state),
            latency=self._state_to_latency(state),
            error_type=state.get("error_type"),
            error_message=state.get("error_message"),
            retrieval_query=state.get("retrieval_query"),
            rewritten_queries=list(state.get("rewritten_queries") or []),
            clarification_question=state.get("clarification_question"),
            context_relevance_score=state.get("context_relevance_score"),
        )
