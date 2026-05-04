"""Unit tests for RAG context, sources, retrieval, rerank, and generation."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from cadence_md.app.enums import VectorSearchType
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag import (
    GENERATE_FALLBACK_ANSWER,
    NO_CONTEXT_ANSWER,
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
from cadence_md.app.settings import settings


def _minimal_qdrant_manager() -> QdrantManager:
    mgr = object.__new__(QdrantManager)
    mgr.search_mode = VectorSearchType.HYBRID
    mgr.hybrid_top_k = 30
    mgr.dense_top_k = 20
    mgr.sparse_top_k = 20
    return mgr


def _make_state(**overrides: Any) -> RAGState:
    """Build a fresh ``RAGState`` with all fields populated to schema defaults."""
    base: RAGState = {
        "query": "q",
        "query_hash": "h",
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

    state = _make_state(query="q1", query_hash="ab")
    out = pipe.retrieve_node(state)
    qm.retrieve.assert_called_once_with("q1")
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
    pipe = RAGPipeline(llm, qm, reranker=reranker)  # type: ignore[arg-type]

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
