"""Tests for shared RAG stack bootstrap helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cadence_md.app.settings import settings
from cadence_md.rag.bootstrap import build_rag_stack
from commands import build_project_cli_parser


@patch("cadence_md.rag.bootstrap.RAGPipeline")
@patch("cadence_md.rag.bootstrap.get_qdrant_manager_from_settings")
@patch("cadence_md.rag.bootstrap.get_llm_from_settings")
@patch("cadence_md.rag.bootstrap.get_reranker_from_settings")
@patch("cadence_md.rag.bootstrap.get_embedder_from_settings")
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

    mock_embedder.assert_called_once_with(settings)
    mock_reranker.assert_called_once_with(settings)
    mock_llm.assert_called_once_with(settings)
    mock_qdrant_factory.assert_called_once_with(mock_embedder.return_value, settings)
    mock_mgr.setup_qdrant.assert_called_once_with()
    mock_rag_cls.assert_called_once_with(
        mock_llm.return_value,
        mock_mgr,
        mock_reranker.return_value,
    )
    assert pipe is mock_pipe
    assert reranker is mock_reranker.return_value


def test_project_cli_rejects_removed_interactive_rag_command() -> None:
    parser = build_project_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["rag"])
