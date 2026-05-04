"""Tests for `cadence_md.app.main` (CLI REPL, RAG stack wiring, state formatting)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from cadence_md.app.main import (
    _format_rag_state,
    build_rag_stack,
    main,
    render_rag_output,
    settings,
)
from cadence_md.app.rag import RAGState


def _minimal_state(**overrides: object) -> RAGState:
    base: RAGState = {
        "query": "Тестовый вопрос",
        "query_hash": "deadbeef",
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
        "answer": "Краткий ответ.",
        "answer_word_count": 2,
        "latency_ms": {"total": 12.5},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def _ranked(
    doc: Document,
    *,
    rank: int = 1,
    retrieval_score: float | None = None,
    rerank_score: float | None = None,
    final_score: float | None = None,
) -> dict[str, object]:
    """Build a ``RankedDocument`` dict for tests."""
    return {
        "rank": rank,
        "doc": doc,
        "retrieval_score": retrieval_score,
        "rerank_score": rerank_score,
        "final_score": final_score,
        "chunk_id": None,
        "section_id": None,
    }


def test_format_rag_state_includes_query_metrics_and_answer() -> None:
    doc = Document(
        page_content="Короткий текст чанка для превью.",
        metadata={"filename": "guide.pdf"},
    )
    state = _minimal_state(
        query="Что делать?",
        rerank_fallback=True,
        context_chars=1200,
        ranked_docs=[_ranked(doc, retrieval_score=0.5, rerank_score=0.88, final_score=0.88)],
        sources=[
            {
                "doc_ref": "[Doc 1]",
                "filename": "guide.pdf",
                "document_title": "Клинреки 2024",
                "section_title": "Терапия",
            }
        ],
    )
    out = _format_rag_state(state)
    assert "Что делать?" in out
    assert "deadbeef" in out
    assert "Rerank fallback:            True" in out
    assert "Клинреки 2024" in out
    assert "section:  Терапия" in out
    assert "guide.pdf" in out
    assert "Краткий ответ." in out
    assert "0.8800" in out
    assert "retrieval=0.5000" in out
    assert "rerank=0.8800" in out
    assert "Документов в контексте:    1" in out


def test_format_rag_state_shows_fallback_flags_and_error() -> None:
    state = _minimal_state(
        retrieval_failed=True,
        generate_fallback=False,
        context_truncated=True,
        error_type="RuntimeError",
        error_message="boom",
    )
    out = _format_rag_state(state)
    assert "Retrieval failed:           True" in out
    assert "Generate fallback:          False" in out
    assert "Context truncated:          True" in out
    assert "Error:" in out and "RuntimeError" in out and "boom" in out


def test_format_rag_state_truncates_long_preview() -> None:
    long_body = "а" * 300
    doc = Document(page_content=long_body, metadata={"filename": "x.pdf"})
    state = _minimal_state(
        ranked_docs=[_ranked(doc, retrieval_score=1.0, rerank_score=1.0, final_score=1.0)],
    )
    out = _format_rag_state(state)
    assert "…" in out
    assert len([ln for ln in out.splitlines() if ln.startswith("     а")]) >= 1


def test_format_rag_state_renders_na_when_final_score_is_none() -> None:
    """A ``None`` ``final_score`` should be displayed as ``n/a`` rather than crashing."""
    doc = Document(page_content="a", metadata={"filename": "1.pdf"})
    state = _minimal_state(
        ranked_docs=[_ranked(doc, retrieval_score=None, rerank_score=None, final_score=None)],
    )
    out = _format_rag_state(state)
    assert "score: n/a (retrieval=n/a, rerank=n/a)" in out


def test_format_rag_state_unknown_filename_metadata() -> None:
    doc = Document(page_content="body", metadata={})
    state = _minimal_state(
        ranked_docs=[_ranked(doc, retrieval_score=0.1, rerank_score=0.1, final_score=0.1)],
    )
    out = _format_rag_state(state)
    assert "unknown" in out


def test_render_rag_output_runnable_matches_format_function() -> None:
    state = _minimal_state()
    assert render_rag_output.invoke(state) == _format_rag_state(state)


@patch("cadence_md.app.main.RAGPipeline")
@patch("cadence_md.app.main.get_qdrant_manager_from_settings")
@patch("cadence_md.app.main.get_llm_from_settings")
@patch("cadence_md.app.main.get_reranker_from_settings")
@patch("cadence_md.app.main.get_embedder_from_settings")
def test_build_rag_stack_calls_setup_and_returns_pipeline(
    mock_embedder: MagicMock,
    mock_reranker: MagicMock,
    mock_llm: MagicMock,
    mock_qdrant_factory: MagicMock,
    mock_rag_cls: MagicMock,
) -> None:
    mock_mgr = MagicMock()
    mock_qdrant_factory.return_value = mock_mgr
    mock_pipe = MagicMock()
    mock_rag_cls.return_value = mock_pipe

    pipe, reranker = build_rag_stack()

    mock_embedder.assert_called_once()
    mock_reranker.assert_called_once()
    mock_llm.assert_called_once()
    mock_qdrant_factory.assert_called_once()
    mock_mgr.setup_qdrant.assert_called_once()
    mock_rag_cls.assert_called_once_with(
        mock_llm.return_value, mock_mgr, mock_reranker.return_value
    )
    mock_qdrant_factory.assert_called_once_with(mock_embedder.return_value, settings)
    assert pipe is mock_pipe
    assert reranker is mock_reranker.return_value


@patch("cadence_md.app.main.build_rag_stack")
@patch("cadence_md.app.main.input", create=True)
@patch("cadence_md.app.main.logging.basicConfig")
def test_main_repl_one_iteration_then_eof(
    _mock_basic_config: MagicMock,
    mock_input: MagicMock,
    mock_build: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_reranker = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _minimal_state(query="один раз")
    mock_build.return_value = (mock_pipeline, mock_reranker)

    mock_input.side_effect = ["игнор", EOFError]

    with pytest.raises(EOFError):
        main()

    mock_pipeline.run.assert_called_once_with("игнор")
    mock_reranker.close.assert_called_once()
    assert "один раз" in capsys.readouterr().out


@patch("cadence_md.app.main.build_rag_stack")
@patch("cadence_md.app.main.input", create=True)
@patch("cadence_md.app.main.logging.basicConfig")
def test_main_eof_before_any_query_closes_reranker(
    _mock_basic_config: MagicMock,
    mock_input: MagicMock,
    mock_build: MagicMock,
) -> None:
    mock_reranker = MagicMock()
    mock_pipeline = MagicMock()
    mock_build.return_value = (mock_pipeline, mock_reranker)
    mock_input.side_effect = EOFError

    with pytest.raises(EOFError):
        main()

    mock_pipeline.run.assert_not_called()
    mock_reranker.close.assert_called_once()


@patch("cadence_md.app.main.build_rag_stack")
@patch("cadence_md.app.main.input", create=True)
@patch("cadence_md.app.main.logging.basicConfig")
def test_main_closes_reranker_when_run_raises(
    _mock_basic_config: MagicMock,
    mock_input: MagicMock,
    mock_build: MagicMock,
) -> None:
    mock_reranker = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.run.side_effect = RuntimeError("pipeline failed")
    mock_build.return_value = (mock_pipeline, mock_reranker)
    mock_input.side_effect = ["вопрос"]

    with pytest.raises(RuntimeError, match="pipeline failed"):
        main()

    mock_reranker.close.assert_called_once()
