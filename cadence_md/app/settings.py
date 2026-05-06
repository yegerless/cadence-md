"""
Pydantic settings and nested RAG configuration.
"""

from pathlib import Path

from fastembed import SparseTextEmbedding
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import models as qdrant_models

from cadence_md.app.enums import QdrantFusionMethod, VectorSearchType


def _default_embedding_query_instruction_path() -> Path:
    """Default UTF-8 file prepended to embedding queries when ``use_query_instruction`` is True."""
    # return (
    #     Path(__file__).resolve().parent / "prompts" / "bge_m3_embedding_query_instruction.txt"
    # )  # For BGE-M3
    return (
        Path(__file__).resolve().parent / "prompts" / "qwen3_embedding_query_instruction.txt"
    )  # For Qwen3-Embedding


def _default_reranker_query_instruction_path() -> Path:
    """Default UTF-8 file for the rerank query prefix when ``use_query_instruction`` is True."""
    return (
        Path(__file__).resolve().parent / "prompts" / "reranker_prompt.txt"
    )  # Only for Qwen3-Reranker


class ChunkConfig(BaseModel):
    """RecursiveCharacterTextSplitter parameters for clinical section documents."""

    chunk_size: int = 2048
    chunk_overlap: int = 256
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", "; ", ", ", " ", ""])


class QdrantConfig(BaseModel):
    """Collection name, vector params, batch size, PDF directory, and FastEmbed sparse model.

    ``sparse_model`` is a FastEmbed sparse embedding instance; ``arbitrary_types_allowed`` lets the
    default BM25 model live in config without a separate env indirection.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_dir: Path = Path("data/main_specialities/")
    collection_name: str = "clinical_recs"
    rebuild_collection: bool = False
    # vector_size: int = 1024  # For BGE-m3 and Qwen3-Embedding-0.6b
    vector_size: int = 2560  # For Qwen3-Embedding-4b
    distance: qdrant_models.Distance = qdrant_models.Distance.COSINE
    uploading_batch_size: int = 128  # 256
    sparse_model: SparseTextEmbedding = Field(
        default_factory=lambda: SparseTextEmbedding(model_name="Qdrant/bm25")
    )


class EmbeddingConfig(BaseModel):
    """Dense embedding model name, optional query instruction file, and HTTP retry policy."""

    # model_name: str = "bge-m3"
    model_name: str = "qwen3-embedding-4b"
    query_instruction_path: Path = Field(default_factory=_default_embedding_query_instruction_path)
    use_query_instruction: bool = True
    normalize_embeddings: bool = True
    return_score: bool = True
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=6, ge=0)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_max_seconds: float = Field(default=120.0, gt=0)


class RerankerConfig(BaseModel):
    """Reranker client: model id, top-k, query instruction file, 429 vs transport retries."""

    # model_name: str = "bge-reranker-v2-m3"
    model_name: str = "qwen3-reranker-4b"
    query_instruction_path: Path = Field(default_factory=_default_reranker_query_instruction_path)
    use_query_instruction: bool = True  # Use only for Qwen3-Reranker
    return_score: bool = True
    top_k: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries_on_rate_limit: int = Field(default=8, ge=0)
    max_retries_on_transport: int = Field(default=3, ge=0)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_max_seconds: float = Field(default=120.0, gt=0)


class RetrievalConfig(BaseModel):
    """Qdrant search mode (dense / sparse / hybrid) and per-mode top-k + fusion method."""

    search_mode: VectorSearchType = VectorSearchType.HYBRID
    fusion_method: QdrantFusionMethod = QdrantFusionMethod.RRF
    sparse_top_k: int = 20
    dense_top_k: int = 20
    hybrid_top_k: int = 15


class LLMConfig(BaseModel):
    """Chat model decoding parameters and HTTP retry policy for :mod:`llm`."""

    model_name: str = "qwen3.5-9b"
    max_new_tokens: int = 5120
    temperature: float = 0.3
    top_p: float = 0.8
    streaming: bool = False
    timeout_seconds: float = Field(default=300.0, gt=0)
    max_retries: int = Field(default=6, ge=0)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_max_seconds: float = Field(default=120.0, gt=0)


class RAGConfig(BaseModel):
    """Full RAG profile: chunking, retrieval, models, and ``prompt_version`` for prompt files."""

    prompt_version: str = Field(
        default="2026-05-03",
        description="Version label for RAG prompt templates (system prompt and telemetry).",
    )
    # Character budget for the assembled LLM context. Whole ``[Doc N]`` blocks are added in rank
    # order until the budget is exhausted; the very first block may be character-truncated to keep
    # the model from receiving an empty context when a single chunk exceeds the budget.
    max_context_chars: int = Field(default=12_000, gt=0)
    chunking: ChunkConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    qdrant_config: QdrantConfig


class Settings(BaseSettings):
    """Top-level app config: Qdrant, inference URLs/keys, nested ``rag_config``."""

    # Environment variables
    QDRANT_BASE_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = Field(..., alias="QDRANT__SERVICE__API_KEY")
    QDRANT_HTTPS: bool = False

    # Model inference settings
    MODEL_INFERENCE_BASE_URL: str = "http://localhost:8080/v1"
    MODEL_INFERENCE_API_KEY: str = "lm-studio"

    # RAG configuration
    rag_config: RAGConfig = Field(
        default_factory=lambda: RAGConfig(
            chunking=ChunkConfig(),
            retrieval=RetrievalConfig(),
            llm=LLMConfig(),
            embedding=EmbeddingConfig(),
            reranker=RerankerConfig(),
            qdrant_config=QdrantConfig(),
        )
    )

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        extra="ignore",
    )


# Eager default for import-time access; override fields via environment in tests or deployment.
settings = Settings()  # type: ignore
