"""Unit tests for RAG context, sources, retrieval, rerank, and generation."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from prometheus_client import REGISTRY

from cadence_md.app.enums import VectorSearchType
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag import (
    GENERATE_FALLBACK_ANSWER,
    NO_CONTEXT_ANSWER,
    OUTPUT_GUARDRAIL_FALLBACK_ANSWER,
    RETRIEVAL_FALLBACK_ANSWER,
    RAGPipeline,
    RAGState,
    _doc_block_header,
    _query_hash,
    _ranked_doc_from_retrieval,
    _rerank_document_text,
    _source_record,
)
from cadence_md.app.reranker import RerankerAPIError, RerankerWrapper
from cadence_md.app.settings import RAGOptionalNodesConfig, settings


def _minimal_qdrant_manager() -> QdrantManager:
    mgr = object.__new__(QdrantManager)
    mgr.search_mode = VectorSearchType.HYBRID
    mgr.hybrid_top_k = 30
    mgr.dense_top_k = 20
    mgr.sparse_top_k = 20
    return mgr


def _optional_nodes_disabled() -> RAGOptionalNodesConfig:
    """Return config that preserves the historical linear pipeline behavior."""
    return RAGOptionalNodesConfig(
        enable_query_rewriter=False,
        enable_context_relevance_grader=False,
        enable_answer_formatter=False,
        enable_output_guardrails=False,
    )


def _make_state(**overrides: Any) -> RAGState:
    """Build a fresh ``RAGState`` with all fields populated to schema defaults."""
    base: RAGState = {
        "query": "q",
        "query_hash": "h",
        "retrieval_query": "q",
        "rewritten_queries": [],
        "clarification_answer": None,
        "allow_clarification": True,
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
    base.update(overrides)  # type: ignore[arg-type]
    return base


def _state_with_retrieved(
    docs: list[Document], scores: list[float | None], **overrides: Any
) -> RAGState:
    """Helper that seeds ``ranked_docs`` from a (docs, scores) pair like ``retrieve_node`` would."""
    ranked = [
        _ranked_doc_from_retrieval(rank=i + 1, doc=doc, score=score)
        for i, (doc, score) in enumerate(zip(docs, scores, strict=True))
    ]
    return _make_state(ranked_docs=ranked, **overrides)


def test_context_node_includes_titles_and_sources() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    doc = Document(
        page_content="Текст рекомендации.",
        metadata={
            "filename": "rec.pdf",
            "document_title": "Рекомендации 2024",
            "section_title": "Лечение",
            "section_id": "sec-1",
        },
    )
    state = _state_with_retrieved([doc], [0.9], query="Вопрос", query_hash="abcd")
    out = pipe.context_node(state)
    assert "[Doc 1]" in out["context"]
    assert "Document title: Рекомендации 2024" in out["context"]
    assert "Section title: Лечение" in out["context"]
    assert out["sources"][0]["doc_ref"] == "[Doc 1]"
    assert out["sources"][0]["section_id"] == "sec-1"
    assert out["context_chars"] == len(out["context"])
    assert out["context_truncated"] is False


def test_reranker_node_fallback_on_rerank_error() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    reranker.top_k = 2
    reranker.rerank.side_effect = RerankerAPIError("down")
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    d1 = Document(page_content="a", metadata={"filename": "1.pdf"})
    d2 = Document(page_content="b", metadata={"filename": "2.pdf"})
    d3 = Document(page_content="c", metadata={"filename": "3.pdf"})
    state = _state_with_retrieved([d1, d2, d3], [0.1, 0.2, 0.3], query_hash="h1")
    out = pipe.reranker_node(state)
    assert out["rerank_fallback"] is True
    assert len(out["ranked_docs"]) == 2
    assert out["ranked_docs"][0]["doc"].page_content == "a"
    assert [rd["rerank_score"] for rd in out["ranked_docs"]] == [None, None]
    assert [rd["final_score"] for rd in out["ranked_docs"]] == [0.1, 0.2]
    reranker.rerank.assert_called_once()
    _, kwargs = reranker.rerank.call_args
    assert "document_texts" in kwargs
    assert len(kwargs["document_texts"]) == 3


def test_generate_node_uses_invoke_messages() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = "Краткий вывод: да [Doc 1]\nПодробнее:\n- пункт [Doc 1]\n"
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    state = _make_state(
        query="Что делать?",
        query_hash="h2",
        context="[Doc 1] Filename: x.pdf\n\nhello",
        context_chars=10,
    )
    out = pipe.generate_node(state)
    llm.invoke.assert_not_called()
    llm.invoke_messages.assert_called_once()
    msgs = llm.invoke_messages.call_args[0][0]
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage)
    assert "ТОЛЬКО" in msgs[0].content or "контекст" in msgs[0].content.lower()
    assert "Что делать?" in msgs[1].content
    assert out["answer_word_count"] > 0
    assert "llm" in out["latency_ms"]


def test_query_hash_matches_sha256_prefix() -> None:
    q = "тест вопроса"
    assert _query_hash(q) == hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]


@pytest.mark.parametrize(
    ("mode", "expected_k"),
    [
        (VectorSearchType.DENSE, 20),
        (VectorSearchType.SPARSE, 20),
        (VectorSearchType.HYBRID, 30),
    ],
)
def test_retrieval_k_for_mode(mode: VectorSearchType, expected_k: int) -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    qm.search_mode = mode
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]
    assert pipe._retrieval_k_for_mode() == expected_k


def test_retrieve_node_calls_qdrant_and_sets_latency() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    d = Document(page_content="x", metadata={"filename": "f.pdf", "section_id": "s"})
    qm.retrieve = MagicMock(return_value=[(d, 0.42)])
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    state = _make_state(query="q1", retrieval_query="rewritten q1", query_hash="ab")
    out = pipe.retrieve_node(state)
    qm.retrieve.assert_called_once_with("rewritten q1")
    assert [rd["doc"] for rd in out["ranked_docs"]] == [d]
    assert [rd["retrieval_score"] for rd in out["ranked_docs"]] == [0.42]
    assert "qdrant" in out["latency_ms"]
    assert out["latency_ms"]["qdrant"] >= 0.0
    assert out["retrieval_failed"] is False
    assert out["ranked_docs"][0]["section_id"] == "s"
    assert out["ranked_docs"][0]["final_score"] == 0.42


def test_retrieve_node_graceful_fallback_on_qdrant_error() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    qm.retrieve = MagicMock(side_effect=RuntimeError("qdrant down"))
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    state = _make_state(query="q-fail", query_hash="fa11")
    out = pipe.retrieve_node(state)

    assert out["retrieval_failed"] is True
    assert out["error_type"] == "RuntimeError"
    assert out["error_message"] == "qdrant down"
    assert out["ranked_docs"] == []
    assert out["sources"] == []
    assert out["context"] == ""
    assert out["context_chars"] == 0
    assert out["answer"] == RETRIEVAL_FALLBACK_ANSWER
    assert out["answer_word_count"] > 0
    assert "qdrant" in out["latency_ms"]
    sample = REGISTRY.get_sample_value(
        "cadence_rag_fallbacks_total",
        labels={"type": "retrieval_failed"},
    )
    assert sample is not None and sample >= 1


def test_reranker_context_generate_short_circuit_on_retrieval_failure() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    state = _make_state(
        retrieval_failed=True,
        error_type="RuntimeError",
        answer=RETRIEVAL_FALLBACK_ANSWER,
        answer_word_count=len(RETRIEVAL_FALLBACK_ANSWER.split()),
    )
    state = pipe.reranker_node(state)
    state = pipe.context_node(state)
    state = pipe.generate_node(state)

    reranker.rerank.assert_not_called()
    llm.invoke_messages.assert_not_called()
    assert state["retrieval_failed"] is True
    assert state["answer"] == RETRIEVAL_FALLBACK_ANSWER
    assert state["context"] == ""


def test_reranker_node_no_docs_short_circuits() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]
    state = _make_state()
    out = pipe.reranker_node(state)
    reranker.rerank.assert_not_called()
    assert out["ranked_docs"] == []
    assert out["rerank_fallback"] is False


def test_reranker_node_success_reorders_and_maps_none_scores() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    d1 = Document(page_content="first", metadata={"filename": "a.pdf"})
    d2 = Document(page_content="second", metadata={"filename": "b.pdf"})
    reranker.rerank.return_value = [(d2, 0.9), (d1, None)]
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]
    state = _state_with_retrieved([d1, d2], [0.1, 0.2])
    out = pipe.reranker_node(state)
    assert out["rerank_fallback"] is False
    assert [rd["doc"].page_content for rd in out["ranked_docs"]] == ["second", "first"]
    assert "rerank" in out["latency_ms"]
    # Retrieval scores must follow the reordered docs, not stay in pre-rerank order.
    assert [rd["retrieval_score"] for rd in out["ranked_docs"]] == [0.2, 0.1]
    assert [rd["rerank_score"] for rd in out["ranked_docs"]] == [0.9, 0.0]
    assert [rd["final_score"] for rd in out["ranked_docs"]] == [0.9, 0.0]


@pytest.mark.parametrize(
    "exc",
    [
        httpx.RequestError("net", request=MagicMock()),
        OSError(5, "io"),
        ValueError("bad payload"),
    ],
)
def test_reranker_node_fallback_on_transient_errors(exc: BaseException) -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    reranker.top_k = 2
    reranker.rerank.side_effect = exc
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]
    d1 = Document(page_content="a", metadata={"filename": "1.pdf"})
    d2 = Document(page_content="b", metadata={"filename": "2.pdf"})
    state = _state_with_retrieved([d1, d2], [0.5, 0.6])
    out = pipe.reranker_node(state)
    assert out["rerank_fallback"] is True
    assert len(out["ranked_docs"]) == 2
    assert [rd["final_score"] for rd in out["ranked_docs"]] == [0.5, 0.6]


def test_context_node_empty_docs() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]
    state = _make_state()
    out = pipe.context_node(state)
    assert out["context"] == ""
    assert out["sources"] == []
    assert out["context_chars"] == 0
    assert out["context_truncated"] is False


def test_context_node_drops_blocks_over_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``max_context_chars`` is small, only whole blocks within budget are included."""
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    # Force a small budget so the second block cannot fit alongside the first one.
    monkeypatch.setattr(settings.rag_config, "max_context_chars", 100)

    body_1 = "alpha " * 5
    body_2 = "beta " * 30
    d1 = Document(page_content=body_1, metadata={"filename": "1.pdf", "section_id": "a"})
    d2 = Document(page_content=body_2, metadata={"filename": "2.pdf", "section_id": "b"})
    state = _state_with_retrieved([d1, d2], [0.9, 0.8])

    out = pipe.context_node(state)

    assert out["context_truncated"] is True
    assert "[Doc 1]" in out["context"]
    assert "[Doc 2]" not in out["context"]
    assert len(out["sources"]) == 1
    # The dropped doc must also be removed from the canonical ranked list so [Doc N] numbering
    # stays aligned across context, sources, and ranked_docs.
    assert len(out["ranked_docs"]) == 1


def test_context_node_truncates_first_block_when_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single oversized block should still produce non-empty context with the header intact."""
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    monkeypatch.setattr(settings.rag_config, "max_context_chars", 80)
    huge_body = "x" * 1000
    d1 = Document(page_content=huge_body, metadata={"filename": "1.pdf"})
    state = _state_with_retrieved([d1], [0.9])

    out = pipe.context_node(state)

    assert out["context_truncated"] is True
    assert out["context"].startswith("[Doc 1]")
    assert len(out["context"]) <= 80
    assert len(out["sources"]) == 1


def test_doc_helpers_and_source_record() -> None:
    doc = Document(
        page_content="body",
        metadata={"filename": "x.pdf", "document_title": "T", "section_id": "s1"},
    )
    assert "[Doc 2]" in _doc_block_header(2, doc)
    assert "Document title: T" in _doc_block_header(2, doc)
    text = _rerank_document_text(doc)
    assert "Filename: x.pdf" in text
    assert "Content:\nbody" in text
    rec = _source_record(3, doc)
    assert rec["doc_ref"] == "[Doc 3]"
    assert rec["section_id"] == "s1"


def test_generate_node_no_context_returns_safe_answer() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    state = _make_state(query="?", query_hash="h")
    out = pipe.generate_node(state)

    llm.invoke_messages.assert_not_called()
    assert out["answer"] == NO_CONTEXT_ANSWER
    assert out["answer_word_count"] > 0
    assert out["latency_ms"]["llm"] == 0.0


def test_generate_node_fallback_on_llm_error() -> None:
    llm = MagicMock()
    llm.invoke_messages.side_effect = RuntimeError("llm down")
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    state = _make_state(
        query="Что делать?",
        query_hash="h",
        context="[Doc 1] Filename: x.pdf\n\nhello",
        context_chars=10,
    )
    out = pipe.generate_node(state)

    llm.invoke_messages.assert_called_once()
    assert out["generate_fallback"] is True
    assert out["error_type"] == "RuntimeError"
    assert out["error_message"] == "llm down"
    assert out["answer"] == GENERATE_FALLBACK_ANSWER
    assert out["answer_word_count"] > 0
    assert "llm" in out["latency_ms"]


def test_run_invokes_full_pipeline_with_mocks() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = "Ответ с цитатой [Doc 1]."
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    doc = Document(page_content="chunk", metadata={"filename": "g.pdf", "section_title": "S"})
    qm.retrieve = MagicMock(return_value=[(doc, 0.9)])
    reranker.rerank.return_value = [(doc, 0.9)]
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        optional_nodes_config=_optional_nodes_disabled(),
    )

    out = pipe.run("Симптомы?")
    assert out["query"] == "Симптомы?"
    assert out["query_hash"] == _query_hash("Симптомы?")
    assert [rd["doc"] for rd in out["ranked_docs"]] == [doc]
    assert out["answer"] == "Ответ с цитатой [Doc 1]."
    assert out["sources"] and out["sources"][0]["filename"] == "g.pdf"
    assert out["ranked_docs"][0]["final_score"] == 0.9
    lat = out["latency_ms"]
    assert "qdrant" in lat and "rerank" in lat and "llm" in lat
    qm.retrieve.assert_called_once()
    reranker.rerank.assert_called_once()
    llm.invoke_messages.assert_called_once()


def test_build_initial_state_has_schema_defaults() -> None:
    state = RAGPipeline.build_initial_state("Симптомы?")
    assert state["query"] == "Симптомы?"
    assert state["query_hash"] == _query_hash("Симптомы?")
    assert state["retrieval_query"] == "Симптомы?"
    assert state["rewritten_queries"] == []
    assert state["clarification_answer"] is None
    assert state["allow_clarification"] is True
    assert state["rewrite_iteration"] == 0
    assert state["query_rewritten"] is False
    assert state["requires_clarification"] is False
    assert state["context_relevance_score"] is None
    assert state["raw_answer"] is None
    assert state["answer_formatted"] is False
    assert state["output_guardrail_iteration"] == 0
    assert state["output_guardrail_passed"] is False
    assert state["output_guardrail_failed"] is False
    assert state["output_guardrail_fallback"] is False
    assert state["output_guardrail_should_retry"] is False
    assert state["max_output_guardrail_iterations_reached"] is False
    assert state["output_guardrail_score"] is None
    assert state["output_guardrail_reason"] is None
    assert state["output_guardrail_unsupported_claims"] == []
    assert state["ranked_docs"] == []
    assert state["sources"] == []
    assert state["context"] == ""
    assert state["answer"] == ""
    assert state["latency_ms"] == {}
    assert state["retrieval_failed"] is False
    assert state["generate_fallback"] is False


def test_build_initial_state_uses_clarification_for_retrieval_query() -> None:
    state = RAGPipeline.build_initial_state(
        "Симптомы?",
        clarification_answer="Пациент взрослый.",
        allow_clarification=False,
    )

    assert state["query"] == "Симптомы?"
    assert state["query_hash"] == _query_hash("Симптомы?")
    assert state["clarification_answer"] == "Пациент взрослый."
    assert state["allow_clarification"] is False
    assert state["retrieval_query"] == "Симптомы?\n\nУточнение пользователя: Пациент взрослый."


def test_query_rewrite_node_keep_and_rewrite() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    llm.invoke_messages.return_value = (
        '{"action":"keep","rewritten_query":"","clarification_question":null,"reason":"ok"}'
    )
    kept = pipe.query_rewrite_node(_make_state(query="q", retrieval_query="q"))
    assert kept["retrieval_query"] == "q"
    assert kept["query_rewritten"] is False

    llm.invoke_messages.return_value = (
        '{"action":"rewrite","rewritten_query":"точный медицинский запрос",'
        '"clarification_question":null,"reason":"better"}'
    )
    rewritten = pipe.query_rewrite_node(_make_state(query="q", retrieval_query="q"))
    assert rewritten["query"] == "q"
    assert rewritten["retrieval_query"] == "точный медицинский запрос"
    assert rewritten["rewritten_queries"] == ["точный медицинский запрос"]
    assert rewritten["rewrite_iteration"] == 1
    assert rewritten["query_rewritten"] is True


def test_query_rewrite_node_clarification_stop_state() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = (
        '{"action":"clarify","rewritten_query":"","clarification_question":'
        '"Уточните диагноз?","reason":"ambiguous"}'
    )
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        optional_nodes_config=RAGOptionalNodesConfig(enable_query_clarification=True),
    )

    out = pipe.query_rewrite_node(_make_state(query="Что делать?"))
    assert out["requires_clarification"] is True
    assert out["clarification_question"] == "Уточните диагноз?"
    assert out["answer"] == "Уточните диагноз?"
    assert out["answer_word_count"] > 0
    assert "query_rewrite" in out["latency_ms"]
    sample = REGISTRY.get_sample_value(
        "cadence_rag_fallbacks_total",
        labels={"type": "clarification_required"},
    )
    assert sample is not None and sample >= 1


def test_query_rewrite_node_disallows_repeated_clarification() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = (
        '{"action":"clarify","rewritten_query":"","clarification_question":'
        '"Уточните диагноз?","reason":"ambiguous"}'
    )
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        optional_nodes_config=RAGOptionalNodesConfig(enable_query_clarification=True),
    )

    out = pipe.query_rewrite_node(
        _make_state(
            query="Что делать?",
            clarification_answer="Пациент взрослый.",
            allow_clarification=False,
        )
    )

    assert out["requires_clarification"] is False
    assert out["clarification_question"] is None
    assert out["retrieval_query"] == "q"


def test_query_rewrite_node_fallback_on_invalid_json() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = "not json"
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    out = pipe.query_rewrite_node(_make_state(query="original", retrieval_query="previous"))
    assert out["query_rewrite_fallback"] is True
    assert out["retrieval_query"] == "original"
    assert "query_rewrite" in out["latency_ms"]
    duration_sample = REGISTRY.get_sample_value(
        "cadence_rag_node_duration_seconds_count",
        labels={"node": "query_rewrite"},
    )
    fallback_sample = REGISTRY.get_sample_value(
        "cadence_rag_fallbacks_total",
        labels={"type": "query_rewrite_fallback"},
    )
    assert duration_sample is not None and duration_sample >= 1
    assert fallback_sample is not None and fallback_sample >= 1


def test_reranker_uses_retrieval_query() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    doc = Document(page_content="a", metadata={"filename": "1.pdf"})
    reranker.rerank.return_value = [(doc, 0.8)]
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    pipe.reranker_node(_state_with_retrieved([doc], [0.5], query="original", retrieval_query="rw"))

    reranker.rerank.assert_called_once()
    assert reranker.rerank.call_args[0][0] == "rw"


def test_context_relevance_routes_generate_or_rewrite() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

    llm.invoke_messages.return_value = (
        '{"is_relevant":true,"score":0.9,"supported_doc_refs":["[Doc 1]"],"reason":"hit"}'
    )
    relevant = pipe.context_relevance_node(
        _make_state(context="[Doc 1] Filename: x.pdf\n\nctx", context_chars=10)
    )
    assert relevant["context_relevance_should_rewrite"] is False
    assert relevant["context_relevance_score"] == 0.9
    assert "context_relevance" in relevant["latency_ms"]

    llm.invoke_messages.return_value = (
        '{"is_relevant":false,"score":0.1,"supported_doc_refs":[],"reason":"miss"}'
    )
    doc = Document(page_content="ctx", metadata={"filename": "x.pdf"})
    retry = pipe.context_relevance_node(
        _state_with_retrieved(
            [doc],
            [0.1],
            context="[Doc 1] Filename: x.pdf\n\nctx",
            context_chars=10,
        )
    )
    assert retry["context_relevance_should_rewrite"] is True
    assert retry["ranked_docs"] == []
    assert retry["context"] == ""


def test_context_relevance_max_rewrite_iterations_reached() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = (
        '{"is_relevant":false,"score":0.1,"supported_doc_refs":[],"reason":"miss"}'
    )
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        optional_nodes_config=RAGOptionalNodesConfig(max_query_rewrite_iterations=1),
    )

    out = pipe.context_relevance_node(
        _make_state(context="[Doc 1] Filename: x.pdf\n\nctx", context_chars=10, rewrite_iteration=1)
    )
    assert out["context_relevance_failed"] is True
    assert out["max_query_rewrite_iterations_reached"] is True
    assert out["context_relevance_should_rewrite"] is False


def test_answer_formatter_success_and_unknown_citation_fallback() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]
    state = _make_state(
        answer="Ответ [Doc 1].",
        context="[Doc 1] Filename: x.pdf\n\nctx",
        sources=[{"doc_ref": "[Doc 1]"}],
    )

    llm.invoke_messages.return_value = (
        '{"formatted_answer":"Краткий вывод:\\nОтвет [Doc 1].\\nЧто важно:\\n- Пункт [Doc 1].'
        "\\nПрактические ориентиры:\\n- Нет.\\nОграничения ответа:\\n- Только контекст.\\n"
        'Источники: [Doc 1]","reason":"formatted"}'
    )
    formatted = pipe.answer_format_node(state)
    assert formatted["raw_answer"] == "Ответ [Doc 1]."
    assert formatted["answer_formatted"] is True
    assert "Что важно:" in formatted["answer"]
    assert "answer_format" in formatted["latency_ms"]

    llm.invoke_messages.return_value = (
        '{"formatted_answer":"Краткий вывод:\\nНовый факт [Doc 2].","reason":"bad"}'
    )
    fallback = pipe.answer_format_node(
        _make_state(answer="Ответ [Doc 1].", sources=[{"doc_ref": "[Doc 1]"}])
    )
    assert fallback["answer"] == "Ответ [Doc 1]."
    assert fallback["answer_format_fallback"] is True


def test_output_guardrails_accepts_grounded_answer() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = (
        '{"is_acceptable":true,"score":0.92,"grounded":true,"citations_valid":true,'
        '"format_ok":true,"unsupported_claims":[],"reason":"grounded"}'
    )
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]
    state = _make_state(
        answer="Ответ подтвержден [Doc 1].",
        context="[Doc 1] Filename: x.pdf\n\nctx",
        context_chars=10,
        sources=[{"doc_ref": "[Doc 1]"}],
    )

    out = pipe.output_guardrails_node(state)

    assert out["answer"] == "Ответ подтвержден [Doc 1]."
    assert out["output_guardrail_passed"] is True
    assert out["output_guardrail_failed"] is False
    assert out["output_guardrail_should_retry"] is False
    assert out["output_guardrail_score"] == 0.92
    assert "output_guardrails" in out["latency_ms"]


def test_output_guardrails_unknown_citation_retries_without_llm() -> None:
    llm = MagicMock()
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]
    doc = Document(page_content="ctx", metadata={"filename": "x.pdf"})
    state = _state_with_retrieved(
        [doc],
        [0.9],
        answer="Неподтвержденная ссылка [Doc 99].",
        answer_word_count=3,
        context="[Doc 1] Filename: x.pdf\n\nctx",
        context_chars=10,
        sources=[{"doc_ref": "[Doc 1]"}],
        raw_answer="raw",
        answer_formatted=True,
        answer_format_fallback=True,
    )

    out = pipe.output_guardrails_node(state)

    llm.invoke_messages.assert_not_called()
    assert out["output_guardrail_failed"] is True
    assert out["output_guardrail_should_retry"] is True
    assert out["output_guardrail_iteration"] == 1
    assert out["ranked_docs"] == []
    assert out["sources"] == []
    assert out["context"] == ""
    assert out["answer"] == ""
    assert out["raw_answer"] is None
    assert out["answer_formatted"] is False
    assert pipe._output_guardrails_route(out) == "query_rewrite"


def test_output_guardrails_rejects_and_falls_back_when_budget_exhausted() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = (
        '{"is_acceptable":false,"score":0.2,"grounded":false,"citations_valid":true,'
        '"format_ok":true,"unsupported_claims":["claim"],"reason":"unsupported"}'
    )
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        optional_nodes_config=RAGOptionalNodesConfig(max_output_guardrail_iterations=0),
    )
    doc = Document(page_content="ctx", metadata={"filename": "x.pdf"})
    state = _state_with_retrieved(
        [doc],
        [0.9],
        answer="Ответ [Doc 1].",
        context="[Doc 1] Filename: x.pdf\n\nctx",
        context_chars=10,
        sources=[{"doc_ref": "[Doc 1]"}],
    )

    out = pipe.output_guardrails_node(state)

    assert out["answer"] == OUTPUT_GUARDRAIL_FALLBACK_ANSWER
    assert out["answer_word_count"] == len(OUTPUT_GUARDRAIL_FALLBACK_ANSWER.split())
    assert out["output_guardrail_failed"] is True
    assert out["output_guardrail_fallback"] is True
    assert out["max_output_guardrail_iterations_reached"] is True
    assert out["output_guardrail_should_retry"] is False
    assert out["sources"] == []
    assert out["context"] == ""
    assert pipe._output_guardrails_route(out) == "end"


def test_output_guardrails_llm_exception_fail_closed() -> None:
    llm = MagicMock()
    llm.invoke_messages.side_effect = RuntimeError("rate limited")
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        optional_nodes_config=RAGOptionalNodesConfig(max_output_guardrail_iterations=0),
    )

    out = pipe.output_guardrails_node(
        _make_state(
            answer="Ответ [Doc 1].",
            context="[Doc 1] Filename: x.pdf\n\nctx",
            context_chars=10,
            sources=[{"doc_ref": "[Doc 1]"}],
        )
    )

    assert out["answer"] == OUTPUT_GUARDRAIL_FALLBACK_ANSWER
    assert out["output_guardrail_failed"] is True
    assert out["output_guardrail_fallback"] is True
    assert out["error_type"] == "RuntimeError"
    assert out["max_output_guardrail_iterations_reached"] is True


def test_default_graph_runs_output_guardrails_after_generate_without_formatter() -> None:
    llm = MagicMock()
    llm.invoke_messages.side_effect = [
        "Ответ [Doc 1].",
        (
            '{"is_acceptable":true,"score":0.9,"grounded":true,"citations_valid":true,'
            '"format_ok":true,"unsupported_claims":[],"reason":"ok"}'
        ),
    ]
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    doc = Document(page_content="chunk", metadata={"filename": "g.pdf"})
    qm.retrieve = MagicMock(return_value=[(doc, 0.9)])
    reranker.rerank.return_value = [(doc, 0.9)]
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        optional_nodes_config=RAGOptionalNodesConfig(
            enable_query_rewriter=False,
            enable_context_relevance_grader=False,
            enable_answer_formatter=False,
            enable_output_guardrails=True,
        ),
    )

    out = pipe.run("Вопрос?")

    assert out["answer"] == "Ответ [Doc 1]."
    assert out["output_guardrail_passed"] is True
    assert "output_guardrails" in out["latency_ms"]
    assert llm.invoke_messages.call_count == 2


def test_default_graph_runs_output_guardrails_after_answer_formatter() -> None:
    llm = MagicMock()
    llm.invoke_messages.side_effect = [
        "Ответ [Doc 1].",
        '{"formatted_answer":"Краткий вывод:\\nОтвет [Doc 1].","reason":"formatted"}',
        (
            '{"is_acceptable":true,"score":0.9,"grounded":true,"citations_valid":true,'
            '"format_ok":true,"unsupported_claims":[],"reason":"ok"}'
        ),
    ]
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    doc = Document(page_content="chunk", metadata={"filename": "g.pdf"})
    qm.retrieve = MagicMock(return_value=[(doc, 0.9)])
    reranker.rerank.return_value = [(doc, 0.9)]
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        optional_nodes_config=RAGOptionalNodesConfig(
            enable_query_rewriter=False,
            enable_context_relevance_grader=False,
            enable_answer_formatter=True,
            enable_output_guardrails=True,
        ),
    )

    out = pipe.run("Вопрос?")

    assert out["raw_answer"] == "Ответ [Doc 1]."
    assert out["answer"] == "Краткий вывод:\nОтвет [Doc 1]."
    assert out["answer_formatted"] is True
    assert out["output_guardrail_passed"] is True
    assert llm.invoke_messages.call_count == 3


def test_disabled_optional_nodes_keep_previous_default_order() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = "Ответ [Doc 1]."
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    doc = Document(page_content="chunk", metadata={"filename": "g.pdf"})
    qm.retrieve = MagicMock(return_value=[(doc, 0.9)])
    reranker.rerank.return_value = [(doc, 0.9)]
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        optional_nodes_config=_optional_nodes_disabled(),
    )

    out = pipe.run("Вопрос?")
    assert out["answer"] == "Ответ [Doc 1]."
    llm.invoke_messages.assert_called_once()
    qm.retrieve.assert_called_once_with("Вопрос?")


def test_run_supports_configured_node_order() -> None:
    llm = MagicMock()
    llm.invoke_messages.return_value = "ok"
    reranker = MagicMock(spec=RerankerWrapper)
    qm = _minimal_qdrant_manager()
    doc = Document(page_content="chunk", metadata={"filename": "g.pdf"})
    qm.retrieve = MagicMock(return_value=[(doc, 0.9)])
    reranker.rerank.return_value = [(doc, 0.9)]
    pipe = RAGPipeline(
        llm,
        qm,
        reranker=reranker,  # type: ignore[arg-type]
        node_order=("retrieve", "context", "generate"),
    )

    out = pipe.run("Вопрос?")
    assert out["answer"] == "ok"
    reranker.rerank.assert_not_called()
