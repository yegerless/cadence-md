from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import models as qdrant_models

from cadence_md.app.enums import QdrantFusionMethod, RerankerAggregationStrategy, VectorSearchType


class ChunkSettings(BaseModel):
    """Text chunking settings for RAG"""

    chunk_size: int = 1024
    chunk_overlap: int = 256


class QdrantSettings(BaseModel):
    """Qdrant connection settings"""

    collection_name: str = "clinical_recs"
    rebuild_collection: bool = False
    vector_size: int = 1024
    distance: qdrant_models.Distance = qdrant_models.Distance.COSINE
    uploading_batch_size: int = 256


class EmbeddingSettings(BaseModel):
    """Embedder model settings"""

    model_name: str = "text-embedding-bge-m3"
    normalize_embeddings: bool = True
    return_score: bool = False


class RerankerSettings(BaseModel):
    """Reranker model settings"""

    model_name: str = "text-embedding-bge-reranker-v2-m3"
    instruction: str | None = None
    embedding_agregation_strategy: RerankerAggregationStrategy = RerankerAggregationStrategy.MAX
    return_score: bool = False
    top_k: int = 5


class RetrievalSettings(BaseModel):
    """Retrieval settings"""

    search_mode: VectorSearchType = VectorSearchType.DENSE
    fusion_method: QdrantFusionMethod = QdrantFusionMethod.RRF
    sparse_top_k: int = 20
    dense_top_k: int = 20
    hybrid_top_k: int = 30


class LLMSettings(BaseModel):
    """LLM settings"""

    model_name: str = "qwen2.5-3b-instruct"
    max_new_tokens: int = 512
    temperature: float = 0.5
    top_p: float = 0.8


class RAGConfig(BaseModel):
    """Configuration for a specific RAG mode"""

    chunking: ChunkSettings
    retrieval: RetrievalSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    qdrant_config: QdrantSettings


class Settings(BaseSettings):
    """Global application settings"""

    # Environment variables
    QDRANT_BASE_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = Field(..., alias="QDRANT__SERVICE__API_KEY")
    QDRANT_HTTPS: bool = False

    # Model inference settings
    MODEL_INFERENCE_BASE_URL: str = "http://localhost:1234/v1"
    MODEL_INFERENCE_API_KEY: str = "lm-studio"

    rag_config: RAGConfig = Field(
        default_factory=lambda: RAGConfig(
            chunking=ChunkSettings(),
            retrieval=RetrievalSettings(),
            llm=LLMSettings(),
            embedding=EmbeddingSettings(),
            reranker=RerankerSettings(),
            qdrant_config=QdrantSettings(),
        )
    )

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        extra="ignore",
    )


# Global settings instance
settings = Settings()  # type: ignore
