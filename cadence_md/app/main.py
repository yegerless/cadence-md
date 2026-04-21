import textwrap

from langchain_core.runnables import RunnableLambda

from cadence_md.app.embedder import get_embedder
from cadence_md.app.llm import get_llm
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag import RAGPipeline, RAGState
from cadence_md.app.reranker import get_reranker
from cadence_md.app.settings import settings

# Init models connection
embedder = get_embedder(
    model=settings.rag_config.embedding.model_name,
    normalize=settings.rag_config.embedding.normalize_embeddings,
    return_score=settings.rag_config.embedding.return_score,
    base_url=settings.MODEL_INFERENCE_BASE_URL,
    api_key=settings.MODEL_INFERENCE_API_KEY,
)
reranker = get_reranker(
    model=settings.rag_config.reranker.model_name,
    instruction=settings.rag_config.reranker.instruction,
    top_k=settings.rag_config.reranker.top_k,
    return_score=settings.rag_config.reranker.return_score,
    embedding_agregation_strategy=settings.rag_config.reranker.embedding_agregation_strategy,
    base_url=settings.MODEL_INFERENCE_BASE_URL,
    api_key=settings.MODEL_INFERENCE_API_KEY,
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

# Init qdrant
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

# Init RAG graph
rag_pipeline = RAGPipeline(llm, qdrant_manager, reranker=reranker)


def _format_rag_state(state: RAGState) -> str:
    """ """
    width = 72
    line = "═" * width
    thin = "─" * width

    blocks: list[str] = [
        line,
        " RAG итоговое состояние",
        line,
        "",
        "Запрос",
        thin,
        state["query"],
        "",
        "Метрики",
        thin,
        f"  Документов после retrieval: {len(state['retrieved_scores'])}",
        f"  Документов после rerank:   {len(state['retrieved_docs'])}",
        f"  Символов в контексте:      {state['context_chars']}",
        f"  Слов в ответе:             {state['answer_word_count']}",
        "",
        f"Документы ({len(state['retrieved_docs'])})",
        thin,
    ]

    scores = state["reranked_scores"]
    for i, doc in enumerate(state["retrieved_docs"], 1):
        score = scores[i - 1] if i - 1 < len(scores) else float("nan")
        name = doc.metadata.get("filename", "unknown")
        preview = (doc.page_content or "").replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:220] + "…"
        blocks.append(f"  {i}. {name}")
        blocks.append(f"     score: {score:.4f}")
        blocks.append(f"     {preview}")
        blocks.append("")

    blocks.extend(
        [
            "Ответ",
            thin,
            textwrap.fill(
                state["answer"],
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            ),
            "",
            line,
            "",
        ]
    )

    return "\n".join(blocks)


# LangChain Runnable for console output
render_rag_output = RunnableLambda(_format_rag_state)


def main() -> None:
    while True:
        query = input("Введите Ваш вопрос:")
        result: RAGState = rag_pipeline.run(query)
        print(render_rag_output.invoke(result))


if __name__ == "__main__":
    main()
