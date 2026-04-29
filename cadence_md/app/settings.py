from pathlib import Path

from fastembed import SparseTextEmbedding
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import models as qdrant_models

from cadence_md.app.enums import QdrantFusionMethod, VectorSearchType


class ChunkConfig(BaseModel):
    """Text chunking settings for RAG"""

    chunk_size: int = 512
    chunk_overlap: int = 256


class QdrantConfig(BaseModel):
    """Qdrant connection settings"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_dir: Path = Path("test_data/")
    collection_name: str = "clinical_recs"
    rebuild_collection: bool = False
    vector_size: int = 1024
    distance: qdrant_models.Distance = qdrant_models.Distance.COSINE
    uploading_batch_size: int = 256
    sparse_model: SparseTextEmbedding = Field(
        default_factory=lambda: SparseTextEmbedding(model_name="Qdrant/bm25")
    )


class EmbeddingConfig(BaseModel):
    """Embedder model settings"""

    model_name: str = "bge-m3"
    normalize_embeddings: bool = True
    return_score: bool = True


class RerankerConfig(BaseModel):
    """Reranker: OpenAI-compatible ``POST /v1/rerank`` (e.g. llama.cpp ``--reranking``)."""

    model_name: str = "bge-reranker-v2-m3"
    return_score: bool = True
    top_k: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries_on_rate_limit: int = Field(default=8, ge=0)


class RetrievalConfig(BaseModel):
    """Retrieval settings"""

    search_mode: VectorSearchType = VectorSearchType.HYBRID
    fusion_method: QdrantFusionMethod = QdrantFusionMethod.RRF
    sparse_top_k: int = 20
    dense_top_k: int = 20
    hybrid_top_k: int = 30


class LLMConfig(BaseModel):
    """LLM settings"""

    model_name: str = "qwen3.5-9b"
    max_new_tokens: int = 5120
    temperature: float = 0.5
    top_p: float = 0.8
    streaming: bool = False


class RAGConfig(BaseModel):
    """Configuration for a specific RAG mode"""

    chunking: ChunkConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    qdrant_config: QdrantConfig


class Settings(BaseSettings):
    """Global application settings"""

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


# Global settings instance
settings = Settings()  # type: ignore
