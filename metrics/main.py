from cadence_md.app.embedder import get_embedder
from cadence_md.app.llm import get_llm
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag import RAGPipeline
from cadence_md.app.reranker import get_reranker
from cadence_md.app.settings import settings
from metrics.config import metrics_settings
from metrics.evaluation_pipeline import RAGEvaluationPipeline
from metrics.gigachat_api_wrapper import ThrottledGigaChat, ThrottledGigaChatEmbeddings


def build_evaluation_pipeline() -> RAGEvaluationPipeline:
    """
    Create and initialize evaluation dependencies lazily

    Returns:
        Evaluation pipeline for RAG
    """
    embedder = get_embedder(
        model=settings.rag_config.embedding.model_name,
        normalize=settings.rag_config.embedding.normalize_embeddings,
        return_score=settings.rag_config.embedding.return_score,
        base_url=settings.MODEL_INFERENCE_BASE_URL,
        api_key=settings.MODEL_INFERENCE_API_KEY,
    )
    reranker = get_reranker(
        model=settings.rag_config.reranker.model_name,
        top_k=settings.rag_config.reranker.top_k,
        return_score=settings.rag_config.reranker.return_score,
        base_url=settings.MODEL_INFERENCE_BASE_URL,
        api_key=settings.MODEL_INFERENCE_API_KEY,
        timeout_s=settings.rag_config.reranker.timeout_seconds,
        max_retries_on_rate_limit=settings.rag_config.reranker.max_retries_on_rate_limit,
    )
    llm = get_llm(
        model=settings.rag_config.llm.model_name,
        base_url=settings.MODEL_INFERENCE_BASE_URL,
        api_key=settings.MODEL_INFERENCE_API_KEY,
        temperature=settings.rag_config.llm.temperature,
        max_completion_tokens=settings.rag_config.llm.max_new_tokens,
        top_p=settings.rag_config.llm.top_p,
        streaming=settings.rag_config.llm.streaming,
    )
    qdrant_manager = QdrantManager(
        data_dir=settings.rag_config.qdrant_config.data_dir,
        chunking_cfg=settings.rag_config.chunking,
        retrieval_cfg=settings.rag_config.retrieval,
        qdrant_cfg=settings.rag_config.qdrant_config,
        url=settings.QDRANT_BASE_URL,
        api_key=settings.QDRANT_API_KEY,
        https=settings.QDRANT_HTTPS,
        embedder=embedder,
        collection_name=settings.rag_config.qdrant_config.collection_name,
        uploading_batch_size=settings.rag_config.qdrant_config.uploading_batch_size,
        sparse_model=settings.rag_config.qdrant_config.sparse_model,
        search_mode=settings.rag_config.retrieval.search_mode,
        fusion_method=settings.rag_config.retrieval.fusion_method,
        sparse_top_k=settings.rag_config.retrieval.sparse_top_k,
        dense_top_k=settings.rag_config.retrieval.dense_top_k,
        hybrid_top_k=settings.rag_config.retrieval.hybrid_top_k,
    )
    qdrant_manager.setup_qdrant()
    rag_pipeline = RAGPipeline(llm, qdrant_manager, reranker=reranker)

    gigachat_llm = ThrottledGigaChat(
        credentials=metrics_settings.GIGACHAT_API_KEY,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        temperature=0.0,
    )
    gigachat_embeddings = ThrottledGigaChatEmbeddings(
        credentials=metrics_settings.GIGACHAT_API_KEY,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="Embeddings",
    )

    return RAGEvaluationPipeline(
        rag_pipeline=rag_pipeline,
        gigachat_llm=gigachat_llm,
        gigachat_embeddings=gigachat_embeddings,
    )
