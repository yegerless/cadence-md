"""Tests for QdrantManager retrieval paths (dense / sparse / hybrid) with mocked client."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from qdrant_client import models as qdrant_models

from cadence_md.app.enums import QdrantFusionMethod, QdrantVectorType, VectorSearchType
from cadence_md.app.qdrant import QdrantManager


def _fake_sparse_embedding() -> MagicMock:
    indices = MagicMock()
    indices.tolist.return_value = [1, 42]
    values = MagicMock()
    values.tolist.return_value = [0.2, 0.8]
    emb = MagicMock()
    emb.indices = indices
    emb.values = values
    return emb


def _retrieval_manager(
    client: MagicMock,
    *,
    search_mode: VectorSearchType,
    fusion_method: QdrantFusionMethod | str = QdrantFusionMethod.RRF,
) -> QdrantManager:
    mgr = object.__new__(QdrantManager)
    mgr.qdrant_client = client
    mgr._collection_name = "clinical_recs"
    mgr.embedder = MagicMock()
    mgr.sparse_model = MagicMock()
    mgr.search_mode = search_mode
    mgr.fusion_method = fusion_method  # type: ignore[assignment]
    mgr.dense_top_k = 5
    mgr.sparse_top_k = 7
    mgr.hybrid_top_k = 10
    return mgr


def test_retrieve_dense_encodes_query_and_maps_payload_to_documents() -> None:
    client = MagicMock()
    pt = MagicMock()
    pt.payload = {"text": "passage text", "filename": "r.pdf", "section_title": "S"}
    pt.score = 0.91
    client.query_points.return_value = MagicMock(points=[pt])

    mgr = _retrieval_manager(client, search_mode=VectorSearchType.DENSE)
    mgr.embedder.encode_query.return_value = [0.0, 1.0]

    pairs = mgr.retrieve("symptoms")

    assert len(pairs) == 1
    doc, score = pairs[0]
    assert doc.page_content == "passage text"
    assert doc.metadata == {"filename": "r.pdf", "section_title": "S"}
    assert score == 0.91
    mgr.embedder.encode_query.assert_called_once_with("symptoms")
    client.query_points.assert_called_once()
    kw = client.query_points.call_args[1]
    assert kw["collection_name"] == "clinical_recs"
    assert kw["using"] == QdrantVectorType.DENSE
    assert kw["limit"] == 5
    assert kw["with_payload"] is True


def test_retrieve_dense_raises_when_encode_fails() -> None:
    client = MagicMock()
    mgr = _retrieval_manager(client, search_mode=VectorSearchType.DENSE)
    mgr.embedder.encode_query.side_effect = ValueError("bad")

    with pytest.raises(RuntimeError, match="Failed to encode query"):
        mgr.retrieve("q")


def test_retrieve_dense_raises_when_query_points_fails() -> None:
    client = MagicMock()
    client.query_points.side_effect = Exception("qdrant down")
    mgr = _retrieval_manager(client, search_mode=VectorSearchType.DENSE)
    mgr.embedder.encode_query.return_value = [0.1]

    with pytest.raises(RuntimeError, match="Failed to query Qdrant"):
        mgr.retrieve("q")


def test_retrieve_sparse_uses_sparse_vector_and_top_k() -> None:
    client = MagicMock()
    pt = MagicMock()
    pt.payload = {"text": "sparse hit"}
    pt.score = 0.5
    client.query_points.return_value = MagicMock(points=[pt])

    fake_emb = _fake_sparse_embedding()
    mgr = _retrieval_manager(client, search_mode=VectorSearchType.SPARSE)
    mgr.sparse_model.embed.return_value = iter([fake_emb])

    pairs = mgr.retrieve("keyword")

    assert len(pairs) == 1
    assert pairs[0][0].page_content == "sparse hit"
    assert pairs[0][1] == 0.5
    kw = client.query_points.call_args[1]
    assert kw["using"] == QdrantVectorType.SPARSE
    assert kw["limit"] == 7
    assert isinstance(kw["query"], qdrant_models.SparseVector)


def test_retrieve_sparse_raises_when_no_sparse_embedding() -> None:
    client = MagicMock()
    mgr = _retrieval_manager(client, search_mode=VectorSearchType.SPARSE)
    mgr.sparse_model.embed.return_value = iter([])

    with pytest.raises(RuntimeError, match="Failed to encode sparse embedding"):
        mgr.retrieve("q")


def test_retrieve_sparse_raises_when_query_points_fails() -> None:
    client = MagicMock()
    client.query_points.side_effect = RuntimeError("timeout")
    mgr = _retrieval_manager(client, search_mode=VectorSearchType.SPARSE)
    mgr.sparse_model.embed.return_value = iter([_fake_sparse_embedding()])

    with pytest.raises(RuntimeError, match="Failed to query Qdrant sparse"):
        mgr.retrieve("q")


@pytest.mark.parametrize(
    "fusion",
    [QdrantFusionMethod.RRF, QdrantFusionMethod.DBSF],
)
def test_retrieve_hybrid_prefetch_and_fusion(
    fusion: QdrantFusionMethod,
) -> None:
    client = MagicMock()
    pt = MagicMock()
    pt.payload = {"text": "hybrid doc"}
    pt.score = 0.77
    client.query_points.return_value = MagicMock(points=[pt])

    mgr = _retrieval_manager(
        client,
        search_mode=VectorSearchType.HYBRID,
        fusion_method=fusion,
    )
    mgr.embedder.encode_query.return_value = [1.0, 2.0]
    mgr.sparse_model.embed.return_value = iter([_fake_sparse_embedding()])

    pairs = mgr.retrieve("clinical question")

    assert len(pairs) == 1
    assert pairs[0][0].page_content == "hybrid doc"
    kw = client.query_points.call_args[1]
    assert len(kw["prefetch"]) == 2
    assert kw["limit"] == 10
    assert isinstance(kw["query"], qdrant_models.FusionQuery)


def test_retrieve_hybrid_raises_on_unknown_fusion_method() -> None:
    client = MagicMock()
    mgr = _retrieval_manager(client, search_mode=VectorSearchType.HYBRID)
    mgr.fusion_method = "not_a_fusion"  # type: ignore[assignment]
    mgr.embedder.encode_query.return_value = [0.1]
    mgr.sparse_model.embed.return_value = iter([_fake_sparse_embedding()])

    with pytest.raises(ValueError, match="Unknown fusion_method"):
        mgr.retrieve("q")


def test_retrieve_hybrid_raises_when_fusion_query_fails() -> None:
    client = MagicMock()
    client.query_points.side_effect = OSError("broken pipe")
    mgr = _retrieval_manager(client, search_mode=VectorSearchType.HYBRID)
    mgr.embedder.encode_query.return_value = [0.1]
    mgr.sparse_model.embed.return_value = iter([_fake_sparse_embedding()])

    with pytest.raises(RuntimeError, match="Failed to perform hybrid search"):
        mgr.retrieve("q")


def test_retrieve_unknown_search_mode_raises() -> None:
    client = MagicMock()
    mgr = _retrieval_manager(client, search_mode=VectorSearchType.DENSE)
    mgr.search_mode = "invalid"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Unknown search_mode"):
        mgr.retrieve("q")


def test_scored_points_none_score_becomes_nan() -> None:
    pt = MagicMock()
    pt.payload = {"text": "x"}
    pt.score = None

    pairs = QdrantManager._scored_points_to_document_score_pairs([pt])

    assert len(pairs) == 1
    assert pairs[0][0].page_content == "x"
    assert math.isnan(pairs[0][1])
