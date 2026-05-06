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


def test_retrieve_uses_retrieve_and_rerank_nodes() -> None:
    pipeline = MagicMock()
    pipeline.build_initial_state.return_value = _base_state(query="query", query_hash="hash-r")
    doc = Document(page_content="a", metadata={"filename": "1.pdf", "section_id": "sec-1"})
    pipeline.retrieve_node.return_value = _base_state(
        query="query",
        query_hash="hash-r",
        ranked_docs=[_ranked_doc(doc, rank=1, retrieval_score=0.4, final_score=0.4)],
        answer="",
    )
    pipeline.reranker_node.return_value = _base_state(
        query="query",
        query_hash="hash-r",
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

    pipeline.build_initial_state.assert_called_once_with("query")
    pipeline.retrieve_node.assert_called_once()
    pipeline.reranker_node.assert_called_once()
    assert response.sources[0].score == 0.8
    assert response.query_hash == "hash-r"
