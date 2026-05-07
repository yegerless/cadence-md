from cadence_md.app.embedder import get_embedder_from_settings
from cadence_md.app.llm import get_llm_from_settings
from cadence_md.app.qdrant import get_qdrant_manager_from_settings
from cadence_md.app.rag import RAGPipeline
from cadence_md.app.reranker import get_reranker_from_settings
from cadence_md.app.settings import settings
from cadence_md.rag import RAGService
from metrics.config import metrics_settings
from metrics.evaluation_pipeline import RAGEvaluationPipeline
from metrics.gigachat_api_wrapper import ThrottledGigaChat, ThrottledGigaChatEmbeddings


def build_evaluation_pipeline(
    optional_nodes_overrides: dict[str, bool] | None = None,
) -> RAGEvaluationPipeline:
    """
    Create and initialize evaluation dependencies lazily

    Returns:
        Evaluation pipeline for RAG
    """
    embedder = get_embedder_from_settings(settings)
    reranker = get_reranker_from_settings(settings)
    llm = get_llm_from_settings(settings)
    qdrant_manager = get_qdrant_manager_from_settings(embedder, settings)
    qdrant_manager.setup_qdrant()
    optional_nodes_config = settings.rag_config.optional_nodes.model_copy(
        update=optional_nodes_overrides or {}
    )
    rag_pipeline = RAGPipeline(
        llm,
        qdrant_manager,
        reranker=reranker,
        optional_nodes_config=optional_nodes_config,
    )
    rag_service = RAGService(pipeline=rag_pipeline)

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
        rag_service=rag_service,
        gigachat_llm=gigachat_llm,
        gigachat_embeddings=gigachat_embeddings,
        rag_optional_nodes_config=optional_nodes_config.model_dump(),
    )
