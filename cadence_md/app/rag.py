"""
LangGraph StateGraph for guideline-grounded answering (retrieve → rerank → generate).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any, TypedDict

import httpx
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from cadence_md.app.enums import VectorSearchType
from cadence_md.app.llm import LLMWrapper
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag_prompts import (
    format_rag_system_prompt,
    load_rag_user_prompt_template,
)
from cadence_md.app.reranker import RerankerAPIError, RerankerWrapper
from cadence_md.app.settings import settings
from cadence_md.observability.langfuse import record_generation, record_span
from cadence_md.observability.metrics import inc_rag_fallback, observe_rag_node
from cadence_md.observability.privacy import redact_medical_query
from cadence_md.observability.settings import observability_settings

logger = logging.getLogger(__name__)


# Safe answers used when the graph has to short-circuit a node because of an unrecoverable error.
# Keep them aligned with the system prompt format ("Краткий вывод:" + "Подробнее:") so downstream
# consumers (CLI, metrics) get a structurally-similar payload.
RETRIEVAL_FALLBACK_ANSWER = (
    "Краткий вывод:\n"
    "В предоставленных фрагментах рекомендаций нет данных: не удалось получить контекст.\n\n"
    "Подробнее:\n"
    "- Поиск по корпусу клинических рекомендаций завершился ошибкой, "
    "поэтому ответ не сформирован.\n"
    "- Попробуйте повторить запрос позже или переформулировать вопрос."
)

GENERATE_FALLBACK_ANSWER = (
    "Краткий вывод:\n"
    "Контекст найден, но генерация ответа LLM не удалась.\n\n"
    "Подробнее:\n"
    "- Релевантные фрагменты рекомендаций получены, но запрос к языковой модели завершился "
    "ошибкой.\n"
    "- Попробуйте повторить запрос позже."
)

NO_CONTEXT_ANSWER = (
    "Краткий вывод:\n"
    "В предоставленных фрагментах рекомендаций нет данных по этому вопросу.\n\n"
    "Подробнее:\n"
    "- Поиск по корпусу не вернул релевантных фрагментов клинических рекомендаций."
)


def _query_hash(query: str) -> str:
    """First 16 hex chars of SHA-256 for log correlation (not a secrecy mechanism)."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _doc_block_header(i: int, doc: Document) -> str:
    """Build ``[Doc i]`` header lines with filename, titles, and ``section_id`` when present."""
    meta = doc.metadata or {}
    lines = [
        f"[Doc {i}]",
        f"Filename: {meta.get('filename', 'unknown')}",
    ]
    if meta.get("document_title"):
        lines.append(f"Document title: {meta['document_title']}")
    if meta.get("section_title"):
        lines.append(f"Section title: {meta['section_title']}")
    if meta.get("section_id"):
        lines.append(f"Section id: {meta['section_id']}")
    return "\n".join(lines)


def _rerank_document_text(doc: Document) -> str:
    """Flatten metadata + body into one string for the cross-encoder / rerank API."""
    meta = doc.metadata or {}
    parts: list[str] = [f"Filename: {meta.get('filename', 'unknown')}"]
    if meta.get("document_title"):
        parts.append(f"Document title: {meta['document_title']}")
    if meta.get("section_title"):
        parts.append(f"Section title: {meta['section_title']}")
    if meta.get("section_id"):
        parts.append(f"Section id: {meta['section_id']}")
    parts.append(f"Content:\n{doc.page_content or ''}")
    return "\n".join(parts)


def _source_record(i: int, doc: Document) -> dict[str, Any]:
    """JSON-friendly citation row matching the ``[Doc i]`` label in the LLM context."""
    meta = doc.metadata or {}
    return {
        "doc_ref": f"[Doc {i}]",
        "filename": meta.get("filename"),
        "source_path": meta.get("source_path"),
        "document_title": meta.get("document_title"),
        "section_title": meta.get("section_title"),
        "section_id": meta.get("section_id"),
    }


class RankedDocument(TypedDict):
    """A single retrieval candidate with all stage scores and stable identifiers.

    ``rank`` is 1-based and must align with the position of the document in
    :attr:`RAGState.ranked_docs`. ``final_score`` is the score that downstream consumers (metrics,
    CLI) should treat as authoritative for the current ordering: it is the rerank score after a
    successful rerank, or the retrieval score on rerank fallback.
    """

    rank: int
    doc: Document
    retrieval_score: float | None
    rerank_score: float | None
    final_score: float | None
    chunk_id: str | None
    section_id: str | None


class RAGState(TypedDict):
    """LangGraph state: retrieval outputs, reranking, assembled prompt context, and telemetry.

    ``ranked_docs`` is the single source of truth for ordered candidates and per-stage scores;
    every consumer (CLI, metrics, tests) reads documents and scores from it directly.
    """

    query: str
    query_hash: str
    ranked_docs: list[RankedDocument]
    rerank_fallback: bool
    retrieval_failed: bool
    generate_fallback: bool
    context_truncated: bool
    error_type: str | None
    error_message: str | None
    sources: list[dict[str, Any]]
    context: str
    context_chars: int
    answer: str
    answer_word_count: int
    latency_ms: dict[str, float]


def _ranked_doc_from_retrieval(rank: int, doc: Document, score: float | None) -> RankedDocument:
    """Build a :class:`RankedDocument` for a freshly retrieved candidate.

    ``rerank_score`` is unset at this stage; ``final_score`` mirrors the retrieval score so the
    record is usable on its own when rerank is bypassed (e.g. zero candidates, fallback path).
    """
    meta = doc.metadata or {}
    return {
        "rank": rank,
        "doc": doc,
        "retrieval_score": score,
        "rerank_score": None,
        "final_score": score,
        "chunk_id": meta.get("chunk_id"),
        "section_id": meta.get("section_id"),
    }


class RAGPipeline:
    """Compile and run a linear RAG graph over :class:`RAGState`.

    The graph is compiled lazily on first :meth:`run` and reused; dependencies are injected so the
    same class works in CLI, tests, and metrics with different LLM/Qdrant/reranker instances.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        qdrant_manager: QdrantManager,
        reranker: RerankerWrapper,
        node_order: tuple[str, ...] | None = None,
    ) -> None:
        self.llm = llm
        self.reranker = reranker
        self.qdrant_manager = qdrant_manager
        self._node_order = node_order or ("retrieve", "rerank", "context", "generate")
        self._graph = None  # Compiled graph; built on first run to avoid import-time graph build.

    @staticmethod
    def build_initial_state(query: str) -> RAGState:
        """Build a fresh initial graph state for a single user query."""
        return {
            "query": query,
            "query_hash": _query_hash(query),
            "ranked_docs": [],
            "rerank_fallback": False,
            "retrieval_failed": False,
            "generate_fallback": False,
            "context_truncated": False,
            "error_type": None,
            "error_message": None,
            "sources": [],
            "context": "",
            "context_chars": 0,
            "answer": "",
            "answer_word_count": 0,
            "latency_ms": {},
        }

    def _retrieval_k_for_mode(self) -> int:
        """Effective ``limit`` for the active vector search mode (dense, sparse, or hybrid cap)."""
        if self.qdrant_manager.search_mode == VectorSearchType.DENSE:
            return self.qdrant_manager.dense_top_k
        if self.qdrant_manager.search_mode == VectorSearchType.SPARSE:
            return self.qdrant_manager.sparse_top_k
        return self.qdrant_manager.hybrid_top_k

    def retrieve_node(self, state: RAGState) -> RAGState:
        """Call Qdrant and store ranked candidates plus retrieval scores; record Qdrant latency.

        On unrecoverable failure (Qdrant unavailable, encoding error, etc.) the node degrades the
        state in-place: ``retrieval_failed=True``, error fields are populated, a safe fallback
        ``answer`` is set, and the rest of the graph short-circuits via per-node guards.
        """
        qh = state["query_hash"]
        k = self._retrieval_k_for_mode()
        t0 = time.perf_counter()
        try:
            retrieved = self.qdrant_manager.retrieve(state["query"])
        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000
            state["ranked_docs"] = []
            state["retrieval_failed"] = True
            state["error_type"] = type(exc).__name__
            state["error_message"] = str(exc)
            state["sources"] = []
            state["context"] = ""
            state["context_chars"] = 0
            state["answer"] = RETRIEVAL_FALLBACK_ANSWER
            state["answer_word_count"] = len(state["answer"].split())
            state.setdefault("latency_ms", {})["qdrant"] = dt_ms
            observe_rag_node("qdrant", dt_ms)
            inc_rag_fallback("retrieval_failed")
            record_span(
                "retrieve",
                metadata={
                    "query_hash": qh,
                    "latency_ms": round(dt_ms, 2),
                    "retrieval_failed": True,
                    "error_type": state["error_type"],
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
            logger.warning(
                "rag.retrieve_fallback",
                extra={
                    "event": "rag.retrieve_fallback",
                    "query_hash": qh,
                    "search_mode": str(self.qdrant_manager.search_mode),
                    "k": k,
                    "latency_ms": round(dt_ms, 2),
                    "error_type": state["error_type"],
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
            return state

        dt_ms = (time.perf_counter() - t0) * 1000

        ranked: list[RankedDocument] = [
            _ranked_doc_from_retrieval(rank=i + 1, doc=doc, score=score)
            for i, (doc, score) in enumerate(retrieved)
        ]
        state["ranked_docs"] = ranked
        state.setdefault("latency_ms", {})["qdrant"] = dt_ms
        observe_rag_node("qdrant", dt_ms)
        record_span(
            "retrieve",
            metadata={
                "query_hash": qh,
                "latency_ms": round(dt_ms, 2),
                "n_docs": len(ranked),
                "search_mode": str(self.qdrant_manager.search_mode),
                "prompt_version": settings.rag_config.prompt_version,
            },
        )

        logger.info(
            "rag.retrieve",
            extra={
                "event": "rag.retrieve",
                "query_hash": qh,
                "search_mode": str(self.qdrant_manager.search_mode),
                "k": k,
                "latency_ms": round(dt_ms, 2),
                "n_docs": len(ranked),
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        return state

    def reranker_node(self, state: RAGState) -> RAGState:
        """Rerank with enriched document text; on failure, slice top-k by retrieval order.

        Skipped when ``retrieval_failed`` is set so a degraded state is not overwritten.
        """
        if state.get("retrieval_failed"):
            return state

        query = state["query"]
        ranked = state.get("ranked_docs") or []
        qh = state["query_hash"]

        if not ranked:
            state["rerank_fallback"] = False
            return state

        docs = [rd["doc"] for rd in ranked]
        retrieval_score_by_id = {id(rd["doc"]): rd["retrieval_score"] for rd in ranked}
        texts = [_rerank_document_text(d) for d in docs]
        t0 = time.perf_counter()
        try:
            pairs = self.reranker.rerank(query, docs, document_texts=texts)
            new_ranked: list[RankedDocument] = []
            for new_rank, (doc, raw_score) in enumerate(pairs, start=1):
                rerank_score = float(raw_score) if raw_score is not None else 0.0
                meta = doc.metadata or {}
                new_ranked.append(
                    {
                        "rank": new_rank,
                        "doc": doc,
                        "retrieval_score": retrieval_score_by_id.get(id(doc)),
                        "rerank_score": rerank_score,
                        "final_score": rerank_score,
                        "chunk_id": meta.get("chunk_id"),
                        "section_id": meta.get("section_id"),
                    }
                )
            state["ranked_docs"] = new_ranked
            state["rerank_fallback"] = False
        except (RerankerAPIError, httpx.RequestError, OSError, ValueError) as exc:
            inc_rag_fallback("rerank_fallback")
            logger.warning(
                "rag.rerank_fallback",
                extra={
                    "event": "rag.rerank_fallback",
                    "query_hash": qh,
                    "error_type": type(exc).__name__,
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
            top_k = self.reranker.top_k
            fallback: list[RankedDocument] = []
            for new_rank, rd in enumerate(ranked[:top_k], start=1):
                fallback.append(
                    {
                        "rank": new_rank,
                        "doc": rd["doc"],
                        "retrieval_score": rd["retrieval_score"],
                        "rerank_score": None,
                        "final_score": rd["retrieval_score"],
                        "chunk_id": rd["chunk_id"],
                        "section_id": rd["section_id"],
                    }
                )
            state["ranked_docs"] = fallback
            state["rerank_fallback"] = True

        dt_ms = (time.perf_counter() - t0) * 1000
        state.setdefault("latency_ms", {})["rerank"] = dt_ms
        observe_rag_node("rerank", dt_ms)
        record_span(
            "rerank",
            metadata={
                "query_hash": qh,
                "latency_ms": round(dt_ms, 2),
                "n_docs": len(state["ranked_docs"]),
                "rerank_fallback": state["rerank_fallback"],
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        logger.info(
            "rag.rerank",
            extra={
                "event": "rag.rerank",
                "query_hash": qh,
                "latency_ms": round(dt_ms, 2),
                "n_docs": len(state["ranked_docs"]),
                "rerank_fallback": state["rerank_fallback"],
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        return state

    def context_node(self, state: RAGState) -> RAGState:
        """Assemble whole ``[Doc N]`` blocks within ``RAGConfig.max_context_chars``.

        Blocks are added in current rank order while the running total stays within budget. If the
        very first block alone exceeds the budget, its body is character-truncated to keep the
        model from receiving an empty context. ``context_truncated`` records both situations so
        downstream telemetry knows the assembled context is incomplete. Skipped when
        ``retrieval_failed`` is set.
        """
        if state.get("retrieval_failed"):
            return state

        t0 = time.perf_counter()
        budget = int(settings.rag_config.max_context_chars)
        separator = "\n\n---\n\n"

        ranked = state.get("ranked_docs") or []
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        included_docs: list[RankedDocument] = []
        truncated = False

        for rd in ranked:
            doc = rd["doc"]
            display_index = len(included_docs) + 1
            header = _doc_block_header(display_index, doc)
            block = f"{header}\n\n{doc.page_content or ''}"

            if not included_docs:
                # First candidate: include it even if it requires body truncation, otherwise the
                # LLM gets empty context when a single chunk exceeds the budget.
                if len(block) <= budget:
                    context_parts.append(block)
                    sources.append(_source_record(display_index, doc))
                    included_docs.append(rd)
                else:
                    header_with_sep = f"{header}\n\n"
                    body_budget = max(0, budget - len(header_with_sep))
                    truncated_body = (doc.page_content or "")[:body_budget]
                    context_parts.append(f"{header}\n\n{truncated_body}")
                    sources.append(_source_record(display_index, doc))
                    included_docs.append(rd)
                    truncated = True
                continue

            current_chars = sum(len(part) for part in context_parts) + len(separator) * (
                len(context_parts)
            )
            projected = current_chars + len(separator) + len(block)
            if projected > budget:
                truncated = True
                break

            context_parts.append(block)
            sources.append(_source_record(display_index, doc))
            included_docs.append(rd)

        # Drop documents that did not make it into the assembled context so ``[Doc N]`` numbering
        # stays aligned across context, sources, and ranked_docs in the final state.
        if len(included_docs) != len(ranked):
            renumbered: list[RankedDocument] = []
            for new_rank, rd in enumerate(included_docs, start=1):
                renumbered.append({**rd, "rank": new_rank})
            state["ranked_docs"] = renumbered

        context = separator.join(context_parts)
        state["context"] = context
        state["context_chars"] = len(context)
        state["sources"] = sources
        state["context_truncated"] = truncated
        dt_ms = (time.perf_counter() - t0) * 1000
        observe_rag_node("context", dt_ms)
        if truncated:
            inc_rag_fallback("context_truncated")
        record_span(
            "context",
            metadata={
                "query_hash": state["query_hash"],
                "latency_ms": round(dt_ms, 2),
                "context_chars": state["context_chars"],
                "n_sources": len(sources),
                "context_truncated": truncated,
                "prompt_version": settings.rag_config.prompt_version,
            },
        )

        logger.info(
            "rag.context",
            extra={
                "event": "rag.context",
                "query_hash": state["query_hash"],
                "context_chars": state["context_chars"],
                "n_sources": len(sources),
                "context_truncated": truncated,
                "max_context_chars": budget,
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        return state

    def generate_node(self, state: RAGState) -> RAGState:
        """Render system+user messages from prompt templates; invoke LLM; log token-ish latency.

        Short-circuits when retrieval failed (the safe answer is already populated). When the LLM
        call fails after wrapper-level retries, falls back to a safe answer and records the error
        on the state so callers can surface the failure mode.
        """
        if state.get("retrieval_failed"):
            return state

        if not state.get("context"):
            state["answer"] = NO_CONTEXT_ANSWER
            state["answer_word_count"] = len(state["answer"].split())
            state.setdefault("latency_ms", {})["llm"] = 0.0
            observe_rag_node("llm", 0.0)
            record_span(
                "generate",
                metadata={
                    "query_hash": state["query_hash"],
                    "latency_ms": 0.0,
                    "skipped": "no_context",
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
            logger.info(
                "rag.generate_skipped_no_context",
                extra={
                    "event": "rag.generate_skipped_no_context",
                    "query_hash": state["query_hash"],
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
            return state

        system = format_rag_system_prompt()
        user_tmpl = load_rag_user_prompt_template()
        user = user_tmpl.format(
            context=state["context"],
            question=state["query"],
        )
        messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=user)]

        t0 = time.perf_counter()
        try:
            answer = self.llm.invoke_messages(messages)
            state["answer"] = answer
        except Exception as exc:
            state["generate_fallback"] = True
            inc_rag_fallback("generate_fallback")
            state["error_type"] = type(exc).__name__
            state["error_message"] = str(exc)
            state["answer"] = GENERATE_FALLBACK_ANSWER
            logger.warning(
                "rag.generate_fallback",
                extra={
                    "event": "rag.generate_fallback",
                    "query_hash": state["query_hash"],
                    "error_type": state["error_type"],
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
        dt_ms = (time.perf_counter() - t0) * 1000
        state.setdefault("latency_ms", {})["llm"] = dt_ms
        state["answer_word_count"] = len(state["answer"].split())
        observe_rag_node("llm", dt_ms)

        lat = state.get("latency_ms", {})
        trace_query = redact_medical_query(
            state["query"],
            observability_settings.LANGFUSE_TRACE_QUERY_MODE,
        )
        trace_output = (
            state["answer"]
            if observability_settings.LANGFUSE_TRACE_QUERY_MODE == "full"
            else "[redacted rag answer]"
        )
        generation_metadata = {
            "query_hash": state["query_hash"],
            "latency_ms_llm": round(dt_ms, 2),
            "model": settings.rag_config.llm.model_name,
            "prompt_version": settings.rag_config.prompt_version,
            "sources": state.get("sources", []),
            "rerank_fallback": state["rerank_fallback"],
            "generate_fallback": state["generate_fallback"],
            "context_truncated": state["context_truncated"],
        }
        record_generation(
            "generate",
            input_data=trace_query,
            output_data=trace_output,
            metadata=generation_metadata,
        )
        record_span("generate", metadata=generation_metadata)
        logger.info(
            "rag.generate",
            extra={
                "event": "rag.generate",
                "query_hash": state["query_hash"],
                "latency_ms_llm": round(dt_ms, 2),
                "latency_ms_qdrant": round(lat.get("qdrant", 0.0), 2),
                "latency_ms_rerank": round(lat.get("rerank", 0.0), 2),
                "context_chars": state["context_chars"],
                "answer_word_count": state["answer_word_count"],
                "rerank_fallback": state["rerank_fallback"],
                "generate_fallback": state["generate_fallback"],
                "context_truncated": state["context_truncated"],
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        return state

    def _build_graph(self):
        """Wire LangGraph nodes in retrieval order and return a compiled graph."""
        workflow = StateGraph(RAGState)

        node_handlers: dict[str, Callable[[RAGState], RAGState]] = {
            "retrieve": self.retrieve_node,
            "rerank": self.reranker_node,
            "context": self.context_node,
            "generate": self.generate_node,
        }
        if not self._node_order:
            raise ValueError("node_order must include at least one node")
        unknown_nodes = [name for name in self._node_order if name not in node_handlers]
        if unknown_nodes:
            raise ValueError(f"Unknown node(s) in node_order: {unknown_nodes}")

        for node_name in self._node_order:
            workflow.add_node(node_name, node_handlers[node_name])

        workflow.add_edge(START, self._node_order[0])
        for src, dst in zip(self._node_order, self._node_order[1:], strict=False):
            workflow.add_edge(src, dst)
        workflow.add_edge(self._node_order[-1], END)

        return workflow.compile()

    def run(self, query: str) -> RAGState:
        """Execute the pipeline for one user query and return the final :class:`RAGState`."""
        self.ensure_compiled()
        return self._graph.invoke(self.build_initial_state(query))

    def ensure_compiled(self) -> None:
        """Compile the LangGraph once without running retrieval or generation."""
        if self._graph is None:
            self._graph = self._build_graph()
