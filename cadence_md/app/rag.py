"""
LangGraph StateGraph for guideline-grounded answering (retrieve → rerank → generate).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable
from itertools import pairwise
from typing import Any, Literal, TypedDict

import httpx
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field, ValidationError

from cadence_md.app.enums import VectorSearchType
from cadence_md.app.llm import LLMWrapper
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag_prompts import (
    format_rag_system_prompt,
    load_answer_formatter_system_prompt,
    load_answer_formatter_user_prompt_template,
    load_context_relevance_system_prompt,
    load_context_relevance_user_prompt_template,
    load_output_guardrails_system_prompt,
    load_output_guardrails_user_prompt_template,
    load_query_rewriter_system_prompt,
    load_query_rewriter_user_prompt_template,
    load_rag_user_prompt_template,
)
from cadence_md.app.reranker import RerankerAPIError, RerankerWrapper
from cadence_md.app.settings import RAGOptionalNodesConfig, settings
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

OUTPUT_GUARDRAIL_FALLBACK_ANSWER = (
    "Краткий вывод:\n"
    "В базе клинических рекомендаций не найден достаточный подтвержденный контекст по вопросу.\n\n"
    "Подробнее:\n"
    "- Ответ невозможно сгенерировать без риска неподтвержденных выводов.\n"
    "- Уточните диагноз, популяцию пациентов, клиническую ситуацию или раздел рекомендации."
)

_DOC_REF_RE = re.compile(r"\[Doc \d+\]")
_TRACE_TEXT_LIMIT = 2_000
_TRACE_DOC_CONTENT_LIMIT = 800


class QueryRewriteDecision(BaseModel):
    """Structured decision returned by the optional query rewriter."""

    action: Literal["rewrite", "keep", "clarify"]
    rewritten_query: str = ""
    clarification_question: str | None = None
    reason: str = ""


class ContextRelevanceGrade(BaseModel):
    """Structured grade returned by the optional context relevance evaluator."""

    is_relevant: bool
    score: float = Field(ge=0.0, le=1.0)
    supported_doc_refs: list[str] = Field(default_factory=list)
    reason: str = ""


class AnswerFormatterResult(BaseModel):
    """Structured result returned by the optional answer formatter."""

    formatted_answer: str
    reason: str = ""


class OutputGuardrailResult(BaseModel):
    """Structured verdict returned by the optional final answer guardrail."""

    is_acceptable: bool
    score: float = Field(ge=0.0, le=1.0)
    grounded: bool
    citations_valid: bool
    format_ok: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""


def _query_hash(query: str) -> str:
    """First 16 hex chars of SHA-256 for log correlation (not a secrecy mechanism)."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _truncate_trace_text(text: str, *, limit: int = _TRACE_TEXT_LIMIT) -> str:
    """Bound text stored in third-party tracing payloads."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def _trace_text(text: str, label: str, *, limit: int = _TRACE_TEXT_LIMIT) -> str:
    """Apply the configured Langfuse text privacy policy to arbitrary RAG text."""
    if observability_settings.LANGFUSE_TRACE_QUERY_MODE == "full":
        return _truncate_trace_text(text, limit=limit)
    text_hash = _query_hash(text or "")
    if observability_settings.LANGFUSE_TRACE_QUERY_MODE == "hash":
        return text_hash
    return f"[redacted {label}: {text_hash}]"


def _trace_query_text(query: str) -> str:
    """Apply the medical-query redaction policy and bound trace payload size."""
    return _truncate_trace_text(
        redact_medical_query(query, observability_settings.LANGFUSE_TRACE_QUERY_MODE)
    )


def _query_with_clarification(query: str, clarification_answer: str | None) -> str:
    """Return the effective clinical question after user clarification."""
    if clarification_answer is None or not clarification_answer.strip():
        return query
    return f"{query}\n\nУточнение пользователя: {clarification_answer.strip()}"


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


def _trace_ranked_doc(rd: RankedDocument) -> dict[str, Any]:
    """Return a compact, privacy-aware document summary for Langfuse."""
    doc = rd["doc"]
    meta = doc.metadata or {}
    payload = {
        "rank": rd["rank"],
        "doc_ref": f"[Doc {rd['rank']}]",
        "filename": meta.get("filename"),
        "document_title": meta.get("document_title"),
        "section_title": meta.get("section_title"),
        "section_id": rd["section_id"],
        "chunk_id": rd["chunk_id"],
        "retrieval_score": rd["retrieval_score"],
        "rerank_score": rd["rerank_score"],
        "final_score": rd["final_score"],
        "content_chars": len(doc.page_content or ""),
    }
    content = doc.page_content or ""
    if content:
        payload["content"] = _trace_text(
            content,
            "document content",
            limit=_TRACE_DOC_CONTENT_LIMIT,
        )
    return {key: value for key, value in payload.items() if value is not None}


def _trace_ranked_docs(ranked_docs: list[RankedDocument]) -> list[dict[str, Any]]:
    """Return ordered document summaries for node input/output payloads."""
    return [_trace_ranked_doc(rd) for rd in ranked_docs]


def _trace_context_payload(state: RAGState) -> dict[str, Any]:
    """Return a privacy-aware summary of the assembled context."""
    context = state.get("context", "")
    payload: dict[str, Any] = {
        "context_chars": state.get("context_chars", len(context)),
        "doc_refs": sorted(_doc_refs(context)),
        "source_count": len(state.get("sources", [])),
    }
    if context:
        payload["context"] = _trace_text(context, "rag context")
    return payload


def _parse_json_model[JsonModelT: BaseModel](
    raw_text: str, model_type: type[JsonModelT]
) -> JsonModelT:
    """Parse an LLM JSON response with LangChain's parser and validate with Pydantic."""
    parser = JsonOutputParser(pydantic_object=model_type)
    parsed = parser.parse(raw_text)
    return model_type.model_validate(parsed)


def _doc_refs(text: str) -> set[str]:
    """Extract citation refs like ``[Doc 1]`` from model output."""
    return set(_DOC_REF_RE.findall(text or ""))


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
    retrieval_query: str
    rewritten_queries: list[str]
    clarification_answer: str | None
    allow_clarification: bool
    rewrite_iteration: int
    query_rewritten: bool
    query_rewrite_fallback: bool
    requires_clarification: bool
    clarification_question: str | None
    context_relevance_score: float | None
    context_relevance_reason: str | None
    context_relevance_supported_doc_refs: list[str]
    context_relevance_failed: bool
    context_relevance_fallback: bool
    context_relevance_should_rewrite: bool
    max_query_rewrite_iterations_reached: bool
    raw_answer: str | None
    answer_formatted: bool
    answer_format_fallback: bool
    output_guardrail_iteration: int
    output_guardrail_passed: bool
    output_guardrail_failed: bool
    output_guardrail_fallback: bool
    output_guardrail_should_retry: bool
    max_output_guardrail_iterations_reached: bool
    output_guardrail_score: float | None
    output_guardrail_reason: str | None
    output_guardrail_unsupported_claims: list[str]
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
        optional_nodes_config: RAGOptionalNodesConfig | None = None,
    ) -> None:
        self.llm = llm
        self.reranker = reranker
        self.qdrant_manager = qdrant_manager
        self._node_order = node_order
        self.optional_nodes_config = optional_nodes_config or settings.rag_config.optional_nodes
        self._graph = None  # Compiled graph; built on first run to avoid import-time graph build.

    @staticmethod
    def build_initial_state(
        query: str,
        *,
        clarification_answer: str | None = None,
        allow_clarification: bool = True,
    ) -> RAGState:
        """Build a fresh initial graph state for a single user query."""
        effective_query = _query_with_clarification(query, clarification_answer)
        return {
            "query": query,
            "query_hash": _query_hash(query),
            "retrieval_query": effective_query,
            "rewritten_queries": [],
            "clarification_answer": clarification_answer,
            "allow_clarification": allow_clarification,
            "rewrite_iteration": 0,
            "query_rewritten": False,
            "query_rewrite_fallback": False,
            "requires_clarification": False,
            "clarification_question": None,
            "context_relevance_score": None,
            "context_relevance_reason": None,
            "context_relevance_supported_doc_refs": [],
            "context_relevance_failed": False,
            "context_relevance_fallback": False,
            "context_relevance_should_rewrite": False,
            "max_query_rewrite_iterations_reached": False,
            "raw_answer": None,
            "answer_formatted": False,
            "answer_format_fallback": False,
            "output_guardrail_iteration": 0,
            "output_guardrail_passed": False,
            "output_guardrail_failed": False,
            "output_guardrail_fallback": False,
            "output_guardrail_should_retry": False,
            "max_output_guardrail_iterations_reached": False,
            "output_guardrail_score": None,
            "output_guardrail_reason": None,
            "output_guardrail_unsupported_claims": [],
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

    def _invoke_json_model[JsonModelT: BaseModel](
        self,
        *,
        node_name: str,
        system_prompt: str,
        user_prompt: str,
        model_type: type[JsonModelT],
    ) -> JsonModelT:
        """Invoke the shared LLM wrapper and parse a typed JSON response."""
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        raw_text = self.llm.invoke_messages(messages)
        try:
            return _parse_json_model(raw_text, model_type)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning(
                "rag.%s_json_parse_failed",
                node_name,
                extra={
                    "event": f"rag.{node_name}_json_parse_failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise

    def query_rewrite_node(self, state: RAGState) -> RAGState:
        """Optionally rewrite the retrieval query while preserving the original user question."""
        cfg = self.optional_nodes_config
        state["context_relevance_should_rewrite"] = False
        state["output_guardrail_should_retry"] = False
        effective_question = _query_with_clarification(
            state["query"],
            state.get("clarification_answer"),
        )
        if not cfg.enable_query_rewriter:
            state["retrieval_query"] = state.get("retrieval_query") or effective_question
            return state

        t0 = time.perf_counter()
        system = load_query_rewriter_system_prompt()
        user_tmpl = load_query_rewriter_user_prompt_template()
        user = user_tmpl.format(
            question=effective_question,
            retrieval_query=state.get("retrieval_query") or effective_question,
            rewrite_iteration=state.get("rewrite_iteration", 0),
            context_relevance_score=state.get("context_relevance_score"),
            context_relevance_reason=state.get("context_relevance_reason") or "",
            output_guardrail_score=(
                ""
                if state.get("output_guardrail_score") is None
                else state["output_guardrail_score"]
            ),
            output_guardrail_reason=state.get("output_guardrail_reason") or "",
            output_guardrail_unsupported_claims=", ".join(
                state.get("output_guardrail_unsupported_claims") or []
            ),
        )
        trace_input = {
            "query": _trace_query_text(state["query"]),
            "effective_question": _trace_query_text(effective_question),
            "retrieval_query": _trace_query_text(
                state.get("retrieval_query") or effective_question
            ),
            "rewrite_iteration": state.get("rewrite_iteration", 0),
            "context_relevance_score": state.get("context_relevance_score"),
            "context_relevance_reason": state.get("context_relevance_reason"),
            "output_guardrail_score": state.get("output_guardrail_score"),
            "output_guardrail_reason": state.get("output_guardrail_reason"),
            "output_guardrail_unsupported_claims": state.get(
                "output_guardrail_unsupported_claims",
                [],
            ),
        }
        try:
            decision = self._invoke_json_model(
                node_name="query_rewrite",
                system_prompt=system,
                user_prompt=user,
                model_type=QueryRewriteDecision,
            )
        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000
            state["query_rewrite_fallback"] = True
            state["retrieval_query"] = effective_question
            state.setdefault("latency_ms", {})["query_rewrite"] = dt_ms
            observe_rag_node("query_rewrite", dt_ms)
            inc_rag_fallback("query_rewrite_fallback")
            record_span(
                "query_rewrite",
                input_data=trace_input,
                output_data={
                    "retrieval_query": _trace_query_text(state["retrieval_query"]),
                    "query_rewrite_fallback": True,
                    "error_type": type(exc).__name__,
                },
                metadata={
                    "query_hash": state["query_hash"],
                    "latency_ms": round(dt_ms, 2),
                    "node": "query_rewrite",
                    "query_rewrite_fallback": True,
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
            logger.warning(
                "rag.query_rewrite_fallback",
                extra={
                    "event": "rag.query_rewrite_fallback",
                    "node": "query_rewrite",
                    "query_hash": state["query_hash"],
                    "latency_ms": round(dt_ms, 2),
                    "error_type": type(exc).__name__,
                    "query_rewrite_fallback": True,
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
            return state

        current_retrieval_query = state.get("retrieval_query") or effective_question
        if (
            decision.action == "clarify"
            and cfg.enable_query_clarification
            and state.get("allow_clarification", True)
        ):
            question = decision.clarification_question or (
                "Уточните, пожалуйста, клинический вопрос для поиска в рекомендациях."
            )
            state["requires_clarification"] = True
            state["clarification_question"] = question
            state["answer"] = question
            state["answer_word_count"] = len(question.split())
            inc_rag_fallback("clarification_required")
        elif decision.action == "rewrite":
            rewritten = decision.rewritten_query.strip()
            if rewritten and rewritten != current_retrieval_query:
                state["retrieval_query"] = rewritten
                state["rewritten_queries"] = [*state.get("rewritten_queries", []), rewritten]
                state["rewrite_iteration"] = int(state.get("rewrite_iteration", 0)) + 1
                state["query_rewritten"] = True
        elif decision.action == "clarify":
            fallback_query = decision.rewritten_query.strip()
            if fallback_query and fallback_query != current_retrieval_query:
                state["retrieval_query"] = fallback_query
                state["rewritten_queries"] = [*state.get("rewritten_queries", []), fallback_query]
                state["rewrite_iteration"] = int(state.get("rewrite_iteration", 0)) + 1
                state["query_rewritten"] = True

        dt_ms = (time.perf_counter() - t0) * 1000
        state.setdefault("latency_ms", {})["query_rewrite"] = dt_ms
        observe_rag_node("query_rewrite", dt_ms)
        record_span(
            "query_rewrite",
            input_data=trace_input,
            output_data={
                "action": decision.action,
                "retrieval_query": _trace_query_text(
                    state.get("retrieval_query") or effective_question
                ),
                "query_rewritten": state["query_rewritten"],
                "requires_clarification": state["requires_clarification"],
                "clarification_question": (
                    _trace_text(state["clarification_question"], "clarification question")
                    if state.get("clarification_question")
                    else None
                ),
                "rewrite_iteration": state["rewrite_iteration"],
                "reason": decision.reason,
            },
            metadata={
                "query_hash": state["query_hash"],
                "latency_ms": round(dt_ms, 2),
                "node": "query_rewrite",
                "action": decision.action,
                "query_rewritten": state["query_rewritten"],
                "requires_clarification": state["requires_clarification"],
                "rewrite_iteration": state["rewrite_iteration"],
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        logger.info(
            "rag.query_rewrite",
            extra={
                "event": "rag.query_rewrite",
                "node": "query_rewrite",
                "query_hash": state["query_hash"],
                "action": decision.action,
                "query_rewritten": state["query_rewritten"],
                "requires_clarification": state["requires_clarification"],
                "rewrite_iteration": state["rewrite_iteration"],
                "latency_ms": round(dt_ms, 2),
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        return state

    def retrieve_node(self, state: RAGState) -> RAGState:
        """Call Qdrant and store ranked candidates plus retrieval scores; record Qdrant latency.

        On unrecoverable failure (Qdrant unavailable, encoding error, etc.) the node degrades the
        state in-place: ``retrieval_failed=True``, error fields are populated, a safe fallback
        ``answer`` is set, and the rest of the graph short-circuits via per-node guards.
        """
        qh = state["query_hash"]
        k = self._retrieval_k_for_mode()
        retrieval_query = state.get("retrieval_query") or state["query"]
        trace_input = {
            "retrieval_query": _trace_query_text(retrieval_query),
            "search_mode": str(self.qdrant_manager.search_mode),
            "k": k,
            "query_rewritten": state.get("query_rewritten", False),
        }
        t0 = time.perf_counter()
        try:
            retrieved = self.qdrant_manager.retrieve(retrieval_query)
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
                input_data=trace_input,
                output_data={
                    "retrieval_failed": True,
                    "error_type": state["error_type"],
                    "n_docs": 0,
                },
                metadata={
                    "query_hash": qh,
                    "latency_ms": round(dt_ms, 2),
                    "retrieval_failed": True,
                    "error_type": state["error_type"],
                    "query_rewritten": state.get("query_rewritten", False),
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
                    "query_rewritten": state.get("query_rewritten", False),
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
            input_data=trace_input,
            output_data={
                "n_docs": len(ranked),
                "documents": _trace_ranked_docs(ranked),
            },
            metadata={
                "query_hash": qh,
                "latency_ms": round(dt_ms, 2),
                "n_docs": len(ranked),
                "search_mode": str(self.qdrant_manager.search_mode),
                "query_rewritten": state.get("query_rewritten", False),
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
                "query_rewritten": state.get("query_rewritten", False),
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

        query = state.get("retrieval_query") or state["query"]
        ranked = state.get("ranked_docs") or []
        qh = state["query_hash"]

        if not ranked:
            state["rerank_fallback"] = False
            return state

        docs = [rd["doc"] for rd in ranked]
        retrieval_score_by_id = {id(rd["doc"]): rd["retrieval_score"] for rd in ranked}
        texts = [_rerank_document_text(d) for d in docs]
        trace_input = {
            "retrieval_query": _trace_query_text(query),
            "n_docs": len(ranked),
            "documents": _trace_ranked_docs(ranked),
        }
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
            input_data=trace_input,
            output_data={
                "n_docs": len(state["ranked_docs"]),
                "rerank_fallback": state["rerank_fallback"],
                "documents": _trace_ranked_docs(state["ranked_docs"]),
            },
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
        trace_input = {
            "max_context_chars": budget,
            "n_candidate_docs": len(ranked),
            "documents": _trace_ranked_docs(ranked),
        }
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
            input_data=trace_input,
            output_data={
                **_trace_context_payload(state),
                "sources": state.get("sources", []),
                "context_truncated": truncated,
            },
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

    def context_relevance_node(self, state: RAGState) -> RAGState:
        """Optionally grade whether the assembled context supports answering the query."""
        cfg = self.optional_nodes_config
        state["context_relevance_should_rewrite"] = False
        if not cfg.enable_context_relevance_grader:
            return state
        if state.get("retrieval_failed") or not state.get("context"):
            return state

        t0 = time.perf_counter()
        system = load_context_relevance_system_prompt()
        user_tmpl = load_context_relevance_user_prompt_template()
        effective_question = _query_with_clarification(
            state["query"],
            state.get("clarification_answer"),
        )
        user = user_tmpl.format(
            question=effective_question,
            retrieval_query=state.get("retrieval_query") or effective_question,
            context=state["context"],
        )
        trace_input = {
            "query": _trace_query_text(state["query"]),
            "effective_question": _trace_query_text(effective_question),
            "retrieval_query": _trace_query_text(
                state.get("retrieval_query") or effective_question
            ),
            **_trace_context_payload(state),
        }
        try:
            grade = self._invoke_json_model(
                node_name="context_relevance",
                system_prompt=system,
                user_prompt=user,
                model_type=ContextRelevanceGrade,
            )
        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000
            state["context_relevance_fallback"] = True
            state["context_relevance_score"] = None
            state["context_relevance_reason"] = f"fallback: {type(exc).__name__}"
            state.setdefault("latency_ms", {})["context_relevance"] = dt_ms
            observe_rag_node("context_relevance", dt_ms)
            inc_rag_fallback("context_relevance_fallback")
            record_span(
                "context_relevance",
                input_data=trace_input,
                output_data={
                    "context_relevance_fallback": True,
                    "error_type": type(exc).__name__,
                    "score": None,
                },
                metadata={
                    "query_hash": state["query_hash"],
                    "latency_ms": round(dt_ms, 2),
                    "node": "context_relevance",
                    "context_relevance_fallback": True,
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
            logger.warning(
                "rag.context_relevance_fallback",
                extra={
                    "event": "rag.context_relevance_fallback",
                    "node": "context_relevance",
                    "query_hash": state["query_hash"],
                    "latency_ms": round(dt_ms, 2),
                    "error_type": type(exc).__name__,
                    "context_relevance_fallback": True,
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )
            return state

        supported_refs = [
            ref for ref in grade.supported_doc_refs if ref in _doc_refs(state.get("context", ""))
        ]
        state["context_relevance_score"] = grade.score
        state["context_relevance_reason"] = grade.reason
        state["context_relevance_supported_doc_refs"] = supported_refs
        is_relevant = (
            grade.is_relevant
            and grade.score >= cfg.context_relevance_min_score
            and len(supported_refs) >= cfg.context_relevance_min_supported_docs
        )

        if not is_relevant:
            can_rewrite = (
                cfg.enable_query_rewriter
                and state.get("rewrite_iteration", 0) < cfg.max_query_rewrite_iterations
            )
            if can_rewrite:
                state["ranked_docs"] = []
                state["sources"] = []
                state["context"] = ""
                state["context_chars"] = 0
                state["context_relevance_should_rewrite"] = True
            else:
                state["context_relevance_failed"] = True
                if state.get("rewrite_iteration", 0) >= cfg.max_query_rewrite_iterations:
                    state["max_query_rewrite_iterations_reached"] = True

        dt_ms = (time.perf_counter() - t0) * 1000
        state.setdefault("latency_ms", {})["context_relevance"] = dt_ms
        observe_rag_node("context_relevance", dt_ms)
        if state["context_relevance_failed"]:
            inc_rag_fallback("context_relevance_failed")
        record_span(
            "context_relevance",
            input_data=trace_input,
            output_data={
                "score": state["context_relevance_score"],
                "reason": state["context_relevance_reason"],
                "is_relevant": is_relevant,
                "should_rewrite": state["context_relevance_should_rewrite"],
                "supported_doc_refs": supported_refs,
                "context_relevance_failed": state["context_relevance_failed"],
            },
            metadata={
                "query_hash": state["query_hash"],
                "latency_ms": round(dt_ms, 2),
                "node": "context_relevance",
                "score": state["context_relevance_score"],
                "is_relevant": is_relevant,
                "should_rewrite": state["context_relevance_should_rewrite"],
                "supported_doc_refs": supported_refs,
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        logger.info(
            "rag.context_relevance",
            extra={
                "event": "rag.context_relevance",
                "node": "context_relevance",
                "query_hash": state["query_hash"],
                "score": state["context_relevance_score"],
                "is_relevant": is_relevant,
                "should_rewrite": state["context_relevance_should_rewrite"],
                "latency_ms": round(dt_ms, 2),
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
                input_data={
                    "query": _trace_query_text(state["query"]),
                    **_trace_context_payload(state),
                },
                output_data={
                    "answer": _trace_text(state["answer"], "rag answer"),
                    "skipped": "no_context",
                },
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
        effective_question = _query_with_clarification(
            state["query"],
            state.get("clarification_answer"),
        )
        user = user_tmpl.format(
            context=state["context"],
            question=effective_question,
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
        record_span(
            "generate",
            input_data={
                "query": trace_query,
                **_trace_context_payload(state),
                "source_count": len(state.get("sources", [])),
            },
            output_data={
                "answer": trace_output,
                "answer_word_count": state["answer_word_count"],
                "generate_fallback": state["generate_fallback"],
            },
            metadata=generation_metadata,
        )
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

    def answer_format_node(self, state: RAGState) -> RAGState:
        """Optionally normalize the generated answer while preserving citations."""
        cfg = self.optional_nodes_config
        if not cfg.enable_answer_formatter:
            return state
        if (
            state.get("retrieval_failed")
            or state.get("requires_clarification")
            or not state.get("answer")
        ):
            return state

        raw_answer = state["answer"]
        state["raw_answer"] = raw_answer
        t0 = time.perf_counter()
        system = load_answer_formatter_system_prompt()
        user_tmpl = load_answer_formatter_user_prompt_template()
        user = user_tmpl.format(
            answer=raw_answer,
            context=state.get("context", ""),
            sources=", ".join(source.get("doc_ref", "") for source in state.get("sources", [])),
        )
        trace_input = {
            "answer": _trace_text(raw_answer, "rag answer"),
            "sources": state.get("sources", []),
            **_trace_context_payload(state),
        }
        try:
            result = self._invoke_json_model(
                node_name="answer_format",
                system_prompt=system,
                user_prompt=user,
                model_type=AnswerFormatterResult,
            )
            formatted = result.formatted_answer.strip()
            allowed_refs = _doc_refs(raw_answer) | {
                source.get("doc_ref", "") for source in state.get("sources", [])
            }
            formatted_refs = _doc_refs(formatted)
            if not formatted or not formatted_refs.issubset(allowed_refs):
                raise ValueError("formatted answer contains unknown citations or is empty")
            state["answer"] = formatted
            state["answer_formatted"] = True
        except Exception as exc:
            state["answer"] = raw_answer
            state["answer_format_fallback"] = True
            inc_rag_fallback("answer_format_fallback")
            logger.warning(
                "rag.answer_format_fallback",
                extra={
                    "event": "rag.answer_format_fallback",
                    "node": "answer_format",
                    "query_hash": state["query_hash"],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "error_type": type(exc).__name__,
                    "answer_format_fallback": True,
                    "prompt_version": settings.rag_config.prompt_version,
                },
            )

        dt_ms = (time.perf_counter() - t0) * 1000
        state.setdefault("latency_ms", {})["answer_format"] = dt_ms
        state["answer_word_count"] = len(state["answer"].split())
        observe_rag_node("answer_format", dt_ms)
        record_span(
            "answer_format",
            input_data=trace_input,
            output_data={
                "answer": _trace_text(state["answer"], "rag answer"),
                "raw_answer_changed": state["answer"] != raw_answer,
                "answer_formatted": state["answer_formatted"],
                "answer_format_fallback": state["answer_format_fallback"],
            },
            metadata={
                "query_hash": state["query_hash"],
                "latency_ms": round(dt_ms, 2),
                "node": "answer_format",
                "answer_formatted": state["answer_formatted"],
                "answer_format_fallback": state["answer_format_fallback"],
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        logger.info(
            "rag.answer_format",
            extra={
                "event": "rag.answer_format",
                "node": "answer_format",
                "query_hash": state["query_hash"],
                "answer_formatted": state["answer_formatted"],
                "answer_format_fallback": state["answer_format_fallback"],
                "latency_ms": round(dt_ms, 2),
                "prompt_version": settings.rag_config.prompt_version,
            },
        )
        return state

    def _output_guardrail_can_retry(self, state: RAGState) -> bool:
        """Return whether the output guardrail still has retry budget."""
        return (
            state.get("output_guardrail_iteration", 0)
            < self.optional_nodes_config.max_output_guardrail_iterations
        )

    @staticmethod
    def _clear_output_guardrail_attempt(state: RAGState) -> None:
        """Clear generated artifacts before retrying retrieval and answer generation."""
        state["ranked_docs"] = []
        state["sources"] = []
        state["context"] = ""
        state["context_chars"] = 0
        state["answer"] = ""
        state["answer_word_count"] = 0
        state["raw_answer"] = None
        state["answer_formatted"] = False
        state["answer_format_fallback"] = False

    @staticmethod
    def _apply_output_guardrail_fallback(state: RAGState) -> None:
        """Replace a defective answer with the safe output-guardrail fallback."""
        state["answer"] = OUTPUT_GUARDRAIL_FALLBACK_ANSWER
        state["answer_word_count"] = len(state["answer"].split())
        state["ranked_docs"] = []
        state["sources"] = []
        state["context"] = ""
        state["context_chars"] = 0
        state["raw_answer"] = None
        state["answer_formatted"] = False
        state["output_guardrail_should_retry"] = False
        state["output_guardrail_fallback"] = True

    def _record_output_guardrails(
        self,
        state: RAGState,
        *,
        trace_input: dict[str, Any],
        dt_ms: float,
        verdict: dict[str, Any],
        error_type: str | None = None,
        level: str = "info",
    ) -> None:
        """Record latency, tracing, and structured logs for the output guardrail node."""
        state.setdefault("latency_ms", {})["output_guardrails"] = dt_ms
        observe_rag_node("output_guardrails", dt_ms)
        record_span(
            "output_guardrails",
            input_data=trace_input,
            output_data={
                **verdict,
                "answer": _trace_text(state.get("answer", ""), "rag answer"),
            },
            metadata={
                "query_hash": state["query_hash"],
                "latency_ms": round(dt_ms, 2),
                "node": "output_guardrails",
                "score": state.get("output_guardrail_score"),
                "passed": state.get("output_guardrail_passed", False),
                "failed": state.get("output_guardrail_failed", False),
                "fallback": state.get("output_guardrail_fallback", False),
                "should_retry": state.get("output_guardrail_should_retry", False),
                "iteration": state.get("output_guardrail_iteration", 0),
                "max_iterations_reached": state.get(
                    "max_output_guardrail_iterations_reached",
                    False,
                ),
                "prompt_version": settings.rag_config.prompt_version,
                **({"error_type": error_type} if error_type else {}),
            },
        )
        logger_fn = logger.warning if level == "warning" else logger.info
        logger_fn(
            "rag.output_guardrails",
            extra={
                "event": "rag.output_guardrails",
                "node": "output_guardrails",
                "query_hash": state["query_hash"],
                "score": state.get("output_guardrail_score"),
                "passed": state.get("output_guardrail_passed", False),
                "failed": state.get("output_guardrail_failed", False),
                "fallback": state.get("output_guardrail_fallback", False),
                "should_retry": state.get("output_guardrail_should_retry", False),
                "iteration": state.get("output_guardrail_iteration", 0),
                "latency_ms": round(dt_ms, 2),
                "error_type": error_type,
                "prompt_version": settings.rag_config.prompt_version,
            },
        )

    def _handle_output_guardrail_rejection(
        self,
        state: RAGState,
        *,
        reason: str,
        score: float | None,
        unsupported_claims: list[str],
        fallback: bool = False,
    ) -> None:
        """Update state for a rejected final answer and either retry or fail closed."""
        state["output_guardrail_passed"] = False
        state["output_guardrail_failed"] = True
        state["output_guardrail_score"] = score
        state["output_guardrail_reason"] = reason
        state["output_guardrail_unsupported_claims"] = unsupported_claims
        state["output_guardrail_fallback"] = fallback

        if self._output_guardrail_can_retry(state):
            state["output_guardrail_iteration"] = (
                int(state.get("output_guardrail_iteration", 0)) + 1
            )
            state["output_guardrail_should_retry"] = True
            self._clear_output_guardrail_attempt(state)
            inc_rag_fallback("output_guardrails_failed")
            if fallback:
                inc_rag_fallback("output_guardrails_fallback")
            return

        state["max_output_guardrail_iterations_reached"] = True
        self._apply_output_guardrail_fallback(state)
        inc_rag_fallback("output_guardrails_failed")
        inc_rag_fallback("output_guardrails_fallback")
        inc_rag_fallback("output_guardrails_max_iterations")

    def output_guardrails_node(self, state: RAGState) -> RAGState:
        """Optionally verify that the final answer is grounded and citation-safe."""
        cfg = self.optional_nodes_config
        state["output_guardrail_should_retry"] = False
        if not cfg.enable_output_guardrails:
            return state
        if (
            state.get("requires_clarification")
            or state.get("retrieval_failed")
            or not state.get("answer")
        ):
            return state

        state["output_guardrail_passed"] = False
        state["output_guardrail_failed"] = False
        state["output_guardrail_fallback"] = False
        answer = state["answer"]
        effective_question = _query_with_clarification(
            state["query"],
            state.get("clarification_answer"),
        )
        trace_input = {
            "query": _trace_query_text(state["query"]),
            "effective_question": _trace_query_text(effective_question),
            "retrieval_query": _trace_query_text(
                state.get("retrieval_query") or effective_question
            ),
            "answer": _trace_text(answer, "rag answer"),
            "sources": state.get("sources", []),
            **_trace_context_payload(state),
        }

        t0 = time.perf_counter()
        allowed_refs = {
            str(source.get("doc_ref"))
            for source in state.get("sources", [])
            if source.get("doc_ref")
        }
        answer_refs = _doc_refs(answer)
        unknown_refs = sorted(answer_refs - allowed_refs)
        if unknown_refs:
            reason = f"unknown citations: {', '.join(unknown_refs)}"
            self._handle_output_guardrail_rejection(
                state,
                reason=reason,
                score=0.0,
                unsupported_claims=[reason],
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            self._record_output_guardrails(
                state,
                trace_input=trace_input,
                dt_ms=dt_ms,
                verdict={
                    "deterministic_citation_check_failed": True,
                    "unknown_refs": unknown_refs,
                    "reason": reason,
                },
                level="warning",
            )
            return state

        system = load_output_guardrails_system_prompt()
        user_tmpl = load_output_guardrails_user_prompt_template()
        user = user_tmpl.format(
            question=effective_question,
            retrieval_query=state.get("retrieval_query") or effective_question,
            context=state.get("context", ""),
            answer=answer,
            sources=", ".join(source.get("doc_ref", "") for source in state.get("sources", [])),
        )
        try:
            result = self._invoke_json_model(
                node_name="output_guardrails",
                system_prompt=system,
                user_prompt=user,
                model_type=OutputGuardrailResult,
            )
        except Exception as exc:
            reason = f"fallback: {type(exc).__name__}"
            state["error_type"] = type(exc).__name__
            state["error_message"] = str(exc)
            self._handle_output_guardrail_rejection(
                state,
                reason=reason,
                score=None,
                unsupported_claims=[],
                fallback=True,
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            self._record_output_guardrails(
                state,
                trace_input=trace_input,
                dt_ms=dt_ms,
                verdict={
                    "output_guardrail_fallback": True,
                    "reason": reason,
                },
                error_type=type(exc).__name__,
                level="warning",
            )
            return state

        state["output_guardrail_score"] = result.score
        state["output_guardrail_reason"] = result.reason
        state["output_guardrail_unsupported_claims"] = result.unsupported_claims
        acceptable = (
            result.is_acceptable
            and result.score >= cfg.output_guardrail_min_score
            and result.grounded
            and result.citations_valid
            and result.format_ok
        )
        if acceptable:
            state["output_guardrail_passed"] = True
            state["output_guardrail_failed"] = False
            state["output_guardrail_fallback"] = False
            state["output_guardrail_should_retry"] = False
        else:
            self._handle_output_guardrail_rejection(
                state,
                reason=result.reason,
                score=result.score,
                unsupported_claims=result.unsupported_claims,
            )

        dt_ms = (time.perf_counter() - t0) * 1000
        self._record_output_guardrails(
            state,
            trace_input=trace_input,
            dt_ms=dt_ms,
            verdict={
                "is_acceptable": result.is_acceptable,
                "accepted": acceptable,
                "grounded": result.grounded,
                "citations_valid": result.citations_valid,
                "format_ok": result.format_ok,
                "unsupported_claims": result.unsupported_claims,
                "reason": result.reason,
            },
            level="warning" if state["output_guardrail_failed"] else "info",
        )
        return state

    @staticmethod
    def _query_rewrite_route(state: RAGState) -> Literal["retrieve", "end"]:
        """Route to retrieval unless the rewriter produced a clarification stop-state."""
        return "end" if state.get("requires_clarification") else "retrieve"

    @staticmethod
    def _context_relevance_route(state: RAGState) -> Literal["query_rewrite", "generate"]:
        """Route back to query rewriting when the relevance grader requests another search."""
        return "query_rewrite" if state.get("context_relevance_should_rewrite") else "generate"

    @staticmethod
    def _output_guardrails_route(state: RAGState) -> Literal["query_rewrite", "end"]:
        """Route back to query rewriting when output guardrails request another attempt."""
        return "query_rewrite" if state.get("output_guardrail_should_retry") else "end"

    def _node_handlers(self) -> dict[str, Callable[[RAGState], RAGState]]:
        """Return all graph node handlers keyed by stable node names."""
        return {
            "query_rewrite": self.query_rewrite_node,
            "retrieve": self.retrieve_node,
            "rerank": self.reranker_node,
            "context": self.context_node,
            "context_relevance": self.context_relevance_node,
            "generate": self.generate_node,
            "answer_format": self.answer_format_node,
            "output_guardrails": self.output_guardrails_node,
        }

    def _build_linear_graph(self, node_order: tuple[str, ...]):
        """Wire a simple linear graph for tests and explicit extension points."""
        workflow = StateGraph(RAGState)
        node_handlers = self._node_handlers()
        if not node_order:
            raise ValueError("node_order must include at least one node")
        unknown_nodes = [name for name in node_order if name not in node_handlers]
        if unknown_nodes:
            raise ValueError(f"Unknown node(s) in node_order: {unknown_nodes}")

        for node_name in node_order:
            workflow.add_node(node_name, node_handlers[node_name])

        workflow.add_edge(START, node_order[0])
        for src, dst in pairwise(node_order):
            workflow.add_edge(src, dst)
        workflow.add_edge(node_order[-1], END)
        return workflow.compile()

    def _build_default_graph(self):
        """Wire the feature-flagged default graph with optional conditional nodes."""
        workflow = StateGraph(RAGState)
        cfg = self.optional_nodes_config
        needs_query_rewrite_node = (
            cfg.enable_query_rewriter
            or cfg.enable_context_relevance_grader
            or cfg.enable_output_guardrails
        )

        if needs_query_rewrite_node:
            workflow.add_node("query_rewrite", self.query_rewrite_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.reranker_node)
        workflow.add_node("context", self.context_node)
        if cfg.enable_context_relevance_grader:
            workflow.add_node("context_relevance", self.context_relevance_node)
        workflow.add_node("generate", self.generate_node)
        if cfg.enable_answer_formatter:
            workflow.add_node("answer_format", self.answer_format_node)
        if cfg.enable_output_guardrails:
            workflow.add_node("output_guardrails", self.output_guardrails_node)

        if cfg.enable_query_rewriter:
            workflow.add_edge(START, "query_rewrite")
            workflow.add_conditional_edges(
                "query_rewrite",
                self._query_rewrite_route,
                {"retrieve": "retrieve", "end": END},
            )
        else:
            workflow.add_edge(START, "retrieve")
            if needs_query_rewrite_node:
                workflow.add_edge("query_rewrite", "retrieve")

        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "context")
        if cfg.enable_context_relevance_grader:
            workflow.add_edge("context", "context_relevance")
            workflow.add_conditional_edges(
                "context_relevance",
                self._context_relevance_route,
                {"query_rewrite": "query_rewrite", "generate": "generate"},
            )
        else:
            workflow.add_edge("context", "generate")
        if cfg.enable_answer_formatter:
            workflow.add_edge("generate", "answer_format")
            final_answer_node = "answer_format"
        else:
            final_answer_node = "generate"
        if cfg.enable_output_guardrails:
            workflow.add_edge(final_answer_node, "output_guardrails")
            workflow.add_conditional_edges(
                "output_guardrails",
                self._output_guardrails_route,
                {"query_rewrite": "query_rewrite", "end": END},
            )
        else:
            workflow.add_edge(final_answer_node, END)
        return workflow.compile()

    def _build_graph(self):
        """Wire LangGraph nodes and return a compiled graph."""
        if self._node_order is not None:
            return self._build_linear_graph(self._node_order)
        return self._build_default_graph()

    def run_retriever_only(
        self,
        query: str,
        *,
        clarification_answer: str | None = None,
        allow_clarification: bool = True,
    ) -> RAGState:
        """Run retrieval-oriented nodes without answer generation or formatting."""
        state = self.build_initial_state(
            query,
            clarification_answer=clarification_answer,
            allow_clarification=allow_clarification,
        )
        state = self.query_rewrite_node(state)
        while not state.get("requires_clarification"):
            state = self.retrieve_node(state)
            state = self.reranker_node(state)
            state = self.context_node(state)
            state = self.context_relevance_node(state)
            if not state.get("context_relevance_should_rewrite"):
                break
            state = self.query_rewrite_node(state)
        return state

    def run(
        self,
        query: str,
        *,
        clarification_answer: str | None = None,
        allow_clarification: bool = True,
    ) -> RAGState:
        """Execute the pipeline for one user query and return the final :class:`RAGState`."""
        self.ensure_compiled()
        return self._graph.invoke(
            self.build_initial_state(
                query,
                clarification_answer=clarification_answer,
                allow_clarification=allow_clarification,
            )
        )

    def ensure_compiled(self) -> None:
        """Compile the LangGraph once without running retrieval or generation."""
        if self._graph is None:
            self._graph = self._build_graph()
