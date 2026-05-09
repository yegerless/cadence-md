"""RAG stack bootstrap shared by worker and offline pipelines."""

from __future__ import annotations

from cadence_md.app.embedder import get_embedder_from_settings
from cadence_md.app.llm import get_llm_from_settings
from cadence_md.app.qdrant import get_qdrant_manager_from_settings
from cadence_md.app.rag import RAGPipeline
from cadence_md.app.reranker import RerankerWrapper, get_reranker_from_settings
from cadence_md.app.settings import settings


def build_rag_stack() -> tuple[RAGPipeline, RerankerWrapper]:
    """Build embedder/reranker/LLM/Qdrant and return pipeline plus reranker.

    Calls ``QdrantManager.setup_qdrant()`` to create or validate collection state
    before constructing ``RAGPipeline``.
    """
    embedder = get_embedder_from_settings(settings)
    reranker = get_reranker_from_settings(settings)
    llm = get_llm_from_settings(settings)
    qdrant_manager = get_qdrant_manager_from_settings(embedder, settings)
    qdrant_manager.setup_qdrant()
    rag_pipeline = RAGPipeline(llm, qdrant_manager, reranker)
    return rag_pipeline, reranker
