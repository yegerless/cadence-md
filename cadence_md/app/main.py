import textwrap
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from cadence_md.app.embedder import get_embedder
from cadence_md.app.llm import get_llm
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag import RAGPipeline, RAGState
from cadence_md.app.reranker import get_reranker

# Init models connection
embedder = get_embedder()
reranker = get_reranker()
llm = get_llm()

# Init qdrant
qdrant_manager = QdrantManager(embedder)
qdrant_manager.setup_qdrant(data_dir=Path("data/clinical_recomendation_pdfs/"))

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
