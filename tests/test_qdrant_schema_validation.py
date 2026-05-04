"""Tests for Qdrant collection schema / metadata validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import models as qdrant_models

from cadence_md.app.enums import QdrantVectorType
from cadence_md.app.qdrant import QdrantManager


def _bare_manager(mock_client: MagicMock) -> QdrantManager:
    mgr = object.__new__(QdrantManager)
    mgr.qdrant_client = mock_client
    mgr._collection_name = "clinical_recs"
    mgr.embedder = MagicMock(model="bge-m3")
    mgr.sparse_model = MagicMock(model_name="Qdrant/bm25")
    qc = MagicMock()
    qc.vector_size = 1024
    qc.distance = qdrant_models.Distance.COSINE
    qc.rebuild_collection = False
    mgr.qdrant_cfg = qc
    return mgr


def _collection_info(*, metadata: dict | None) -> MagicMock:
    """``get_collection``-shaped mock: metadata lives on ``config.metadata`` (REST/UI)."""
    dense = MagicMock(size=1024, distance=qdrant_models.Distance.COSINE)
    sparse = MagicMock(modifier=qdrant_models.Modifier.IDF)
    params = MagicMock()
    params.vectors = {QdrantVectorType.DENSE: dense}
    params.sparse_vectors = {QdrantVectorType.SPARSE: sparse}
    config = MagicMock(params=params, metadata=metadata)
    info = MagicMock(config=config)
    info.metadata = None
    return info


def _named_collection(name: str) -> SimpleNamespace:
    """Qdrant collection list entry with a real ``name`` attribute (not MagicMock's ``name`` kw)."""
    return SimpleNamespace(name=name)


def test_ensure_raises_when_collection_missing() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="does not exist"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_passes_matching_schema_and_metadata() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    meta = {
        "embedding_model": "bge-m3",
        "dense_vector_size": "1024",
        "distance": str(qdrant_models.Distance.COSINE),
        "sparse_modifier": str(qdrant_models.Modifier.IDF),
        "sparse_model": "Qdrant/bm25",
    }
    client.get_collection.return_value = _collection_info(metadata=meta)
    client.count.return_value = MagicMock(count=10)

    mgr = _bare_manager(client)
    mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_raises_on_vector_size_mismatch() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    dense = MagicMock(size=512, distance=qdrant_models.Distance.COSINE)
    sparse = MagicMock(modifier=qdrant_models.Modifier.IDF)
    params = MagicMock()
    params.vectors = {QdrantVectorType.DENSE: dense}
    params.sparse_vectors = {QdrantVectorType.SPARSE: sparse}
    config = MagicMock(params=params, metadata=None)
    info = MagicMock(config=config)
    info.metadata = None
    client.get_collection.return_value = info
    client.count.return_value = MagicMock(count=1)

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="vector size mismatch"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_writes_metadata_on_empty_collection_without_metadata() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    info_before = _collection_info(metadata=None)
    meta = {
        "embedding_model": "bge-m3",
        "dense_vector_size": "1024",
        "distance": str(qdrant_models.Distance.COSINE),
        "sparse_modifier": str(qdrant_models.Modifier.IDF),
        "sparse_model": "Qdrant/bm25",
    }
    info_after = _collection_info(metadata=meta)
    client.get_collection.side_effect = [info_before, info_before, info_after]
    client.count.return_value = MagicMock(count=0)

    mgr = _bare_manager(client)
    mgr.ensure_collection_exists_and_schema_matches()
    client.update_collection.assert_called_once()
    call_kw = client.update_collection.call_args[1]
    assert call_kw["collection_name"] == "clinical_recs"
    assert call_kw["metadata"]["embedding_model"] == "bge-m3"


def test_ensure_raises_when_list_collections_fails() -> None:
    client = MagicMock()
    client.get_collections.side_effect = ConnectionError("refused")

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="Cannot list Qdrant collections"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_raises_when_dense_vector_missing() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    sparse = MagicMock(modifier=qdrant_models.Modifier.IDF)
    params = MagicMock()
    params.vectors = {}
    params.sparse_vectors = {QdrantVectorType.SPARSE: sparse}
    config = MagicMock(params=params, metadata=None)
    info = MagicMock(config=config)
    info.metadata = None
    client.get_collection.return_value = info

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="missing dense vector"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_raises_when_sparse_vector_missing() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    dense = MagicMock(size=1024, distance=qdrant_models.Distance.COSINE)
    params = MagicMock()
    params.vectors = {QdrantVectorType.DENSE: dense}
    params.sparse_vectors = {}
    config = MagicMock(params=params, metadata=None)
    info = MagicMock(config=config)
    info.metadata = None
    client.get_collection.return_value = info

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="missing sparse vector"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_raises_on_distance_mismatch() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    dense = MagicMock(size=1024, distance=qdrant_models.Distance.DOT)
    sparse = MagicMock(modifier=qdrant_models.Modifier.IDF)
    params = MagicMock()
    params.vectors = {QdrantVectorType.DENSE: dense}
    params.sparse_vectors = {QdrantVectorType.SPARSE: sparse}
    config = MagicMock(params=params, metadata=None)
    info = MagicMock(config=config)
    info.metadata = None
    client.get_collection.return_value = info

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="distance mismatch"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_raises_on_sparse_modifier_mismatch() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    dense = MagicMock(size=1024, distance=qdrant_models.Distance.COSINE)
    sparse = MagicMock(modifier=qdrant_models.Modifier.NONE)
    params = MagicMock()
    params.vectors = {QdrantVectorType.DENSE: dense}
    params.sparse_vectors = {QdrantVectorType.SPARSE: sparse}
    config = MagicMock(params=params, metadata=None)
    info = MagicMock(config=config)
    info.metadata = None
    client.get_collection.return_value = info

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="Sparse vector modifier mismatch"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_raises_when_points_exist_but_metadata_missing() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    info = _collection_info(metadata=None)
    client.get_collection.return_value = info
    client.count.return_value = MagicMock(count=3)

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="has points but no cadence metadata"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_raises_when_metadata_still_missing_after_empty_count_failure() -> None:
    """Count failure yields -1 points; treated as non-empty → cannot auto-repair metadata."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    info = _collection_info(metadata=None)
    client.get_collection.return_value = info
    client.count.side_effect = RuntimeError("count unavailable")

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="has points but no cadence metadata"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_passes_when_metadata_only_on_info_root() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    dense = MagicMock(size=1024, distance=qdrant_models.Distance.COSINE)
    sparse = MagicMock(modifier=qdrant_models.Modifier.IDF)
    params = MagicMock()
    params.vectors = {QdrantVectorType.DENSE: dense}
    params.sparse_vectors = {QdrantVectorType.SPARSE: sparse}
    meta = {
        "embedding_model": "bge-m3",
        "dense_vector_size": "1024",
        "distance": str(qdrant_models.Distance.COSINE),
        "sparse_modifier": str(qdrant_models.Modifier.IDF),
        "sparse_model": "Qdrant/bm25",
    }
    config = MagicMock(params=params, metadata=None)
    info = MagicMock(config=config)
    info.metadata = meta
    client.get_collection.return_value = info

    mgr = _bare_manager(client)
    mgr.ensure_collection_exists_and_schema_matches()


def test_ensure_raises_on_metadata_key_mismatch() -> None:
    client = MagicMock()
    client.get_collections.return_value = MagicMock(
        collections=[_named_collection("clinical_recs")]
    )
    meta = {
        "embedding_model": "other-model",
        "dense_vector_size": "1024",
        "distance": str(qdrant_models.Distance.COSINE),
        "sparse_modifier": str(qdrant_models.Modifier.IDF),
        "sparse_model": "Qdrant/bm25",
    }
    client.get_collection.return_value = _collection_info(metadata=meta)
    client.count.return_value = MagicMock(count=1)

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="Collection metadata mismatch"):
        mgr.ensure_collection_exists_and_schema_matches()


def test_write_collection_metadata_raises_on_update_failure() -> None:
    client = MagicMock()
    client.update_collection.side_effect = Exception("forbidden")

    mgr = _bare_manager(client)
    with pytest.raises(RuntimeError, match="Failed to write collection metadata"):
        mgr._write_collection_schema_metadata()


def test_sparse_model_descriptor_falls_back_to_class_name() -> None:
    class _SparseStub:
        pass

    client = MagicMock()
    mgr = _bare_manager(client)
    mgr.sparse_model = _SparseStub()

    assert mgr._sparse_model_descriptor() == "_SparseStub"
