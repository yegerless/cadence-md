import os
from dataclasses import dataclass, field

import torch
from dotenv import load_dotenv
from qdrant_client import models as qdrant_models

from cadence_md.app.enums import QdrantFusionMethod, RerankerAggregationStrategy, VectorSearchType

load_dotenv(dotenv_path=".env.dev", override=True)
QDRANT_API_KEY = os.getenv("QDRANT__SERVICE__API_KEY", "")

MODEL_INFERENCE_BASE_URL = "http://localhost:1234/v1"
MODEL_INFERENCE_API_KEY = "lm-studio"


@dataclass
class ChunkingConfig:
    chunk_size: int = 1024
    chunk_overlap: int = 256


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "clinical_recs"
    rebuild_collection: bool = True
    vector_size: int = 1024
    distance = qdrant_models.Distance.COSINE
    api_key: str = QDRANT_API_KEY
    https: bool = False
    uploading_batch_size: int = 256


@dataclass
class EmbeddingConfig:
    model_name: str = "text-embedding-bge-m3"
    normalize_embeddings: bool = True
    return_score: bool = False


@dataclass
class RerankerConfig:
    model_name: str = "text-embedding-bge-reranker-v2-m3"
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
    embedding_agregation_strategy: RerankerAggregationStrategy = RerankerAggregationStrategy.MAX
    return_score: bool = False
    top_k: int = 5


@dataclass
class RetrievalConfig:
    search_mode: VectorSearchType = VectorSearchType.HYBRID
    fusion_method: QdrantFusionMethod = QdrantFusionMethod.RRF
    sparse_top_k: int = 20
    dense_top_k: int = 20
    hybrid_top_k: int = 20
    # bm25_k1: float = 1.5
    # bm25_b: float = 0.75


@dataclass
class LLMConfig:
    model_name: str = "qwen2.5-3b-instruct"
    dtype: torch.dtype = torch.float16
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.2


@dataclass
class RAGConfig:
    name: str = "baseline"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    qdrant_config: QdrantConfig = field(default_factory=QdrantConfig)
