"""Tests for Qdrant manager factory from settings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cadence_md.app.qdrant import get_qdrant_manager_from_settings


@patch("cadence_md.app.qdrant.QdrantClient")
def test_get_qdrant_manager_from_settings(mock_client_class: MagicMock, settings) -> None:
    mock_client_class.return_value = MagicMock()
    embedder = MagicMock()
    embedder.model = "bge-m3"

    mgr = get_qdrant_manager_from_settings(embedder, settings)

    qc = settings.rag_config.qdrant_config
    retr = settings.rag_config.retrieval
    assert mgr.search_mode == retr.search_mode
    assert mgr.fusion_method == retr.fusion_method
    assert mgr._collection_name == qc.collection_name
    assert mgr.sparse_top_k == retr.sparse_top_k
    assert mgr.dense_top_k == retr.dense_top_k
    assert mgr.hybrid_top_k == retr.hybrid_top_k
    assert mgr.uploading_batch_size == qc.uploading_batch_size
    assert mgr.embedder is embedder
    mock_client_class.assert_called_once_with(
        url=settings.QDRANT_BASE_URL,
        api_key=settings.QDRANT_API_KEY,
        https=settings.QDRANT_HTTPS,
    )


@patch("cadence_md.app.qdrant.QdrantClient", side_effect=OSError("tls handshake failed"))
def test_get_qdrant_manager_from_settings_raises_when_client_init_fails(
    _mock_client_class: MagicMock,
    settings,
) -> None:
    embedder = MagicMock()
    embedder.model = "bge-m3"

    with pytest.raises(RuntimeError, match="Cannot connect to Qdrant"):
        get_qdrant_manager_from_settings(embedder, settings)
