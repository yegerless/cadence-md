import os
from dataclasses import dataclass, field

import torch
from dotenv import load_dotenv
from qdrant_client import models as qdrant_models

load_dotenv(dotenv_path="../.env.dev", override=True)
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
    model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True
    normalize_embeddings: bool = True
    return_score: bool = False
    max_length: int = 512
    return_dense: bool = True
    return_sparse: bool = False


@dataclass
class RerankerConfig:
    model_name: str = ""


@dataclass
class RetrievalConfig:
    top_k: int = 5
    overfetch_k: int = 5


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
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
    qdrant_config: QdrantConfig = field(default_factory=QdrantConfig)
