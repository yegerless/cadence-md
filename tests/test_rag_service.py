"""Unit tests for stable RAG contract/service adapters."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from cadence_md.rag import RAGRequest, RAGService


def _ranked_doc(
    doc: Document,
    *,
    rank: int = 1,
    retrieval_score: float | None = None,
    rerank_score: float | None = None,
    final_score: float | None = None,
) -> dict:
    return {
        "rank": rank,
        "doc": doc,
        "retrieval_score": retrieval_score,
        "rerank_score": rerank_score,
        "final_score": final_score,
        "chunk_id": doc.metadata.get("chunk_id"),
        "section_id": doc.metadata.get("section_id"),
    }


def _base_state(**overrides: object) -> dict:
    state = {
        "query": "q",
        "query_hash": "hash1",
        "retrieval_query": "q",
        "rewritten_queries": [],
        "query_rewritten": False,
        "query_rewrite_fallback": False,
        "requires_clarification": False,
        "clarification_question": None,
        "context_relevance_score": None,
        "context_relevance_failed": False,
        "context_relevance_fallback": False,
        "max_query_rewrite_iterations_reached": False,
        "raw_answer": None,
        "answer_formatted": False,
        "answer_format_fallback": False,
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
        "answer": "answer",
        "answer_word_count": 1,
        "latency_ms": {"qdrant": 12.0, "rerank": 7.0, "llm": 55.0},
    }
    state.update(overrides)
    return state


def test_run_maps_ranked_docs_to_sources_with_score_order() -> None:
    pipeline = MagicMock()
    doc1 = Document(
        page_content="a",
        metadata={
            "filename": "1.pdf",
            "source_path": "main_specialities/1.pdf",
            "section_id": "s1",
        },
    )
    doc2 = Document(page_content="b", metadata={"filename": "2.pdf", "section_id": "s2"})
    pipeline.run.return_value = _base_state(
        ranked_docs=[
            _ranked_doc(doc1, rank=1, retrieval_score=0.2, rerank_score=0.9, final_score=0.9),
            _ranked_doc(doc2, rank=2, retrieval_score=0.1, rerank_score=0.7, final_score=0.7),
        ]
    )
    service = RAGService(pipeline=pipeline)

    response = service.run(RAGRequest(query="query"))

    assert [src.filename for src in response.sources] == ["1.pdf", "2.pdf"]
    assert response.sources[0].source_path == "main_specialities/1.pdf"
    assert [src.doc_ref for src in response.sources] == ["[Doc 1]", "[Doc 2]"]
    assert [src.score for src in response.sources] == [0.9, 0.7]
    assert response.latency.total_ms == 74.0


def test_run_preserves_flags_and_errors_on_retrieval_failed() -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = _base_state(
        retrieval_failed=True,
        error_type="RuntimeError",
        error_message="qdrant down",
        answer="fallback",
    )
    service = RAGService(pipeline=pipeline)

    response = service.run(RAGRequest(query="query"))

    assert response.flags.retrieval_failed is True
    assert response.error_type == "RuntimeError"
    assert response.error_message == "qdrant down"
    assert response.answer == "fallback"


def test_run_maps_generate_fallback_and_context_truncated_flags() -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = _base_state(generate_fallback=True, context_truncated=True)
    service = RAGService(pipeline=pipeline)

    response = service.run(RAGRequest(query="query"))

    assert response.flags.generate_fallback is True
    assert response.flags.context_truncated is True


def test_run_maps_optional_fields_and_flags() -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = _base_state(
        retrieval_query="rewritten",
        rewritten_queries=["rewritten"],
        query_rewritten=True,
        query_rewrite_fallback=True,
        requires_clarification=True,
        clarification_question="Уточните?",
        context_relevance_score=0.2,
        context_relevance_failed=True,
        context_relevance_fallback=True,
        max_query_rewrite_iterations_reached=True,
        raw_answer="raw",
        answer_formatted=True,
        answer_format_fallback=True,
    )
    service = RAGService(pipeline=pipeline)

    response = service.run(RAGRequest(query="query"))

    assert response.retrieval_query == "rewritten"
    assert response.rewritten_queries == ["rewritten"]
    assert response.clarification_question == "Уточните?"
    assert response.context_relevance_score == 0.2
    assert response.raw_answer == "raw"
    assert response.flags.query_rewritten is True
    assert response.flags.query_rewrite_fallback is True
    assert response.flags.requires_clarification is True
    assert response.flags.context_relevance_failed is True
    assert response.flags.context_relevance_fallback is True
    assert response.flags.max_query_rewrite_iterations_reached is True
    assert response.flags.answer_formatted is True
    assert response.flags.answer_format_fallback is True


def test_run_handles_empty_context_and_empty_docs() -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = _base_state(
        ranked_docs=[],
        context="",
        context_chars=0,
        answer_word_count=0,
        answer="",
    )
    service = RAGService(pipeline=pipeline)

    response = service.run(RAGRequest(query="query"))

    assert response.sources == []
    assert response.context_chars == 0
    assert response.answer == ""


def test_run_keeps_none_final_score() -> None:
    pipeline = MagicMock()
    doc = Document(page_content="a", metadata={"filename": "1.pdf"})
    pipeline.run.return_value = _base_state(
        ranked_docs=[
            _ranked_doc(
                doc,
                rank=1,
                retrieval_score=0.5,
                rerank_score=None,
                final_score=None,
            )
        ]
    )
    service = RAGService(pipeline=pipeline)

    response = service.run(RAGRequest(query="query"))

    assert len(response.sources) == 1
    assert response.sources[0].score is None


def test_retrieve_uses_retriever_only_helper() -> None:
    pipeline = MagicMock()
    doc = Document(page_content="a", metadata={"filename": "1.pdf", "section_id": "sec-1"})
    pipeline.run_retriever_only.return_value = _base_state(
        query="query",
        query_hash="hash-r",
        retrieval_query="query rewritten",
        rewritten_queries=["query rewritten"],
        query_rewritten=True,
        context_relevance_score=0.9,
        ranked_docs=[
            _ranked_doc(
                doc,
                rank=1,
                retrieval_score=0.4,
                rerank_score=0.8,
                final_score=0.8,
            )
        ],
        answer="",
    )
    service = RAGService(pipeline=pipeline)

    response = service.retrieve(RAGRequest(query="query"))

    pipeline.run_retriever_only.assert_called_once_with("query")
    assert response.sources[0].score == 0.8
    assert response.query_hash == "hash-r"
    assert response.retrieval_query == "query rewritten"
    assert response.rewritten_queries == ["query rewritten"]
    assert response.flags.query_rewritten is True
    assert response.context_relevance_score == 0.9
