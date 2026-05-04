"""Unit tests for :mod:`cadence_md.app.settings` (Pydantic settings and nested RAG config)."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import pytest
from pydantic import ValidationError
from qdrant_client import models as qdrant_models

from cadence_md.app.enums import QdrantFusionMethod, VectorSearchType
from cadence_md.app.settings import (
    ChunkConfig,
    EmbeddingConfig,
    LLMConfig,
    QdrantConfig,
    RAGConfig,
    RerankerConfig,
    RetrievalConfig,
    Settings,
    _default_embedding_query_instruction_path,
    _default_reranker_query_instruction_path,
)


def _settings(**kwargs: object) -> Settings:
    """Build :class:`Settings` without reading ``.env.dev`` (stable tests)."""
    base: dict[str, object] = {"QDRANT__SERVICE__API_KEY": "test-api-key", "_env_file": None}
    base.update(kwargs)
    return Settings(**base)  # type: ignore[call-arg]


class TestDefaultInstructionPaths:
    def test_default_embedding_query_instruction_path(self) -> None:
        path = _default_embedding_query_instruction_path()
        assert path.name == "bge_m3_embedding_query_instruction.txt"
        assert path.is_file()

    def test_default_reranker_query_instruction_path(self) -> None:
        path = _default_reranker_query_instruction_path()
        assert path.name == "reranker_prompt.txt"
        assert path.is_file()


class TestChunkConfig:
    def test_defaults(self) -> None:
        c = ChunkConfig()
        assert c.chunk_size == 2048
        assert c.chunk_overlap == 256
        assert "\n\n" in c.separators


class TestSettingsRequiredAndTopLevel:
    def test_missing_qdrant_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("QDRANT__SERVICE__API_KEY", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)  # type: ignore[call-arg]
        errs = exc_info.value.errors()
        assert any(e["loc"] == ("QDRANT__SERVICE__API_KEY",) for e in errs)

    def test_api_key_alias_sets_attribute(self) -> None:
        s = _settings(QDRANT__SERVICE__API_KEY="secret-from-alias")
        assert s.QDRANT_API_KEY == "secret-from-alias"

    def test_top_level_defaults(self) -> None:
        s = _settings()
        assert s.QDRANT_BASE_URL == "http://localhost:6333"
        assert s.QDRANT_HTTPS is False
        assert s.MODEL_INFERENCE_BASE_URL == "http://localhost:8080/v1"
        assert s.MODEL_INFERENCE_API_KEY == "lm-studio"

    def test_top_level_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT__SERVICE__API_KEY", "k")
        monkeypatch.setenv("QDRANT_BASE_URL", "http://qdrant.example:6333")
        monkeypatch.setenv("QDRANT_HTTPS", "true")
        monkeypatch.setenv("MODEL_INFERENCE_BASE_URL", "http://infer.example/v1")
        monkeypatch.setenv("MODEL_INFERENCE_API_KEY", "infer-secret")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.QDRANT_BASE_URL == "http://qdrant.example:6333"
        assert s.QDRANT_HTTPS is True
        assert s.MODEL_INFERENCE_BASE_URL == "http://infer.example/v1"
        assert s.MODEL_INFERENCE_API_KEY == "infer-secret"

    def test_extra_env_vars_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QDRANT__SERVICE__API_KEY", "k")
        monkeypatch.setenv("CADENCE_UNKNOWN_SETTING_XYZ", "should-be-ignored")
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert not hasattr(s, "CADENCE_UNKNOWN_SETTING_XYZ")


class TestRAGConfigNestedDefaults:
    def test_default_rag_config_tree(self) -> None:
        s = _settings()
        rc = s.rag_config
        assert rc.prompt_version == "2026-05-03"
        assert rc.max_context_chars == 12_000
        assert rc.chunking.chunk_size == 2048
        assert rc.retrieval.search_mode == VectorSearchType.HYBRID
        assert rc.retrieval.fusion_method == QdrantFusionMethod.RRF
        assert rc.llm.model_name == "qwen3.5-9b"
        assert rc.llm.temperature == 0.3
        assert rc.embedding.model_name == "bge-m3"
        assert rc.embedding.use_query_instruction is True
        assert rc.reranker.model_name == "bge-reranker-v2-m3"
        assert rc.reranker.use_query_instruction is False
        assert rc.reranker.top_k == 5
        assert rc.qdrant_config.collection_name == "clinical_recs"
        assert rc.qdrant_config.vector_size == 1024
        assert rc.qdrant_config.distance == qdrant_models.Distance.COSINE

    def test_embedding_default_instruction_path_factory(self) -> None:
        s = _settings()
        p = s.rag_config.embedding.query_instruction_path
        assert isinstance(p, Path)
        assert p.name == "bge_m3_embedding_query_instruction.txt"

    def test_explicit_rag_config_override(self) -> None:
        rc = RAGConfig(
            prompt_version="custom-v",
            max_context_chars=4096,
            chunking=ChunkConfig(chunk_size=512, chunk_overlap=64),
            retrieval=RetrievalConfig(
                search_mode=VectorSearchType.DENSE,
                sparse_top_k=10,
                dense_top_k=20,
                hybrid_top_k=15,
            ),
            llm=LLMConfig(model_name="other-llm", temperature=0.9),
            embedding=EmbeddingConfig(model_name="custom-emb", use_query_instruction=False),
            reranker=RerankerConfig(model_name="custom-rank", top_k=12, use_query_instruction=True),
            qdrant_config=QdrantConfig(collection_name="other_collection", vector_size=768),
        )
        s = _settings(rag_config=rc)
        assert s.rag_config.prompt_version == "custom-v"
        assert s.rag_config.max_context_chars == 4096
        assert s.rag_config.chunking.chunk_size == 512
        assert s.rag_config.retrieval.search_mode == VectorSearchType.DENSE
        assert s.rag_config.llm.model_name == "other-llm"
        assert s.rag_config.embedding.model_name == "custom-emb"
        assert s.rag_config.reranker.top_k == 12
        assert s.rag_config.qdrant_config.collection_name == "other_collection"
        assert s.rag_config.qdrant_config.vector_size == 768

    @pytest.mark.parametrize("invalid", [0, -1, -42_000])
    def test_max_context_chars_must_be_positive(self, invalid: int) -> None:
        with pytest.raises(ValidationError):
            RAGConfig(
                max_context_chars=invalid,
                chunking=ChunkConfig(),
                retrieval=RetrievalConfig(),
                llm=LLMConfig(),
                embedding=EmbeddingConfig(),
                reranker=RerankerConfig(),
                qdrant_config=QdrantConfig(),
            )


class TestFieldValidators:
    def test_embedding_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(timeout_seconds=0)

    def test_embedding_backoff_bounds(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(backoff_base_seconds=0)

    def test_embedding_max_retries_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(max_retries=-1)

    def test_reranker_top_k_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            RerankerConfig(top_k=0)

    def test_llm_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(timeout_seconds=-1.0)

    def test_llm_max_retries_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(max_retries=-1)


class TestRetrievalEnums:
    def test_search_mode_from_string(self) -> None:
        r = RetrievalConfig.model_validate({"search_mode": "sparse"})
        assert r.search_mode == VectorSearchType.SPARSE

    def test_fusion_method_from_string(self) -> None:
        r = RetrievalConfig.model_validate({"fusion_method": "dbsf"})
        assert r.fusion_method == QdrantFusionMethod.DBSF


class TestEnvFileNotLoadedWhenDisabled:
    def test_env_file_none_ignores_cwd_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # If ``.env.dev`` in cwd were merged, it could override; ``_env_file=None`` disables files.
        monkeypatch.setenv("QDRANT__SERVICE__API_KEY", "from-env")
        (tmp_path / ".env.dev").write_text(
            "QDRANT__SERVICE__API_KEY=from-dotenv-file\n", encoding="utf-8"
        )
        with chdir(tmp_path):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.QDRANT_API_KEY == "from-env"
