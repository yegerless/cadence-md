"""
Interactive terminal entrypoint for the CADENCE-MD RAG stack.
"""

from __future__ import annotations

import logging
import sys
import textwrap

from langchain_core.runnables import RunnableLambda

from cadence_md.app.embedder import get_embedder_from_settings
from cadence_md.app.llm import get_llm_from_settings
from cadence_md.app.qdrant import get_qdrant_manager_from_settings
from cadence_md.app.rag import RAGPipeline, RAGState
from cadence_md.app.reranker import RerankerWrapper, get_reranker_from_settings
from cadence_md.app.settings import settings

logger = logging.getLogger(__name__)


def build_rag_stack() -> tuple[RAGPipeline, RerankerWrapper]:
    """Wire embedder → reranker → LLM → Qdrant and return the runnable pipeline plus reranker.

    Calls :meth:`cadence_md.app.qdrant.QdrantManager.setup_qdrant`, which may create or validate
    the hybrid collection and optionally re-index PDFs when ``rebuild_collection`` is True.

    Returns:
        ``RAGPipeline`` for :meth:`~cadence_md.app.rag.RAGPipeline.run`, and ``RerankerWrapper``
        so the caller can :meth:`~cadence_md.app.reranker.RerankerWrapper.close` its HTTP client.
    """
    embedder = get_embedder_from_settings(settings)
    reranker = get_reranker_from_settings(settings)
    llm = get_llm_from_settings(settings)

    qdrant_manager = get_qdrant_manager_from_settings(embedder, settings)
    qdrant_manager.setup_qdrant()

    rag_pipeline = RAGPipeline(llm, qdrant_manager, reranker)
    return rag_pipeline, reranker


def _format_rag_state(state: RAGState) -> str:
    """Format :class:`~cadence_md.app.rag.RAGState` for terminal display (Russian labels).

    Shows query hash, retrieval/rerank sizes, latency breakdown, citation-aligned sources, reranked
    document previews, and the final answer wrapped with :func:`textwrap.fill`.
    """
    width = 72
    line = "═" * width
    thin = "─" * width

    ranked_docs = state.get("ranked_docs") or []
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
        f"  query_hash:                {state['query_hash']}",
        f"  Документов в контексте:    {len(ranked_docs)}",
        f"  Rerank fallback:            {state['rerank_fallback']}",
        f"  Retrieval failed:           {state.get('retrieval_failed', False)}",
        f"  Generate fallback:          {state.get('generate_fallback', False)}",
        f"  Context truncated:          {state.get('context_truncated', False)}",
        f"  Символов в контексте:      {state['context_chars']}",
        f"  Слов в ответе:             {state['answer_word_count']}",
        f"  Latency (ms):               {state.get('latency_ms', {})}",
    ]
    error_type = state.get("error_type")
    if error_type:
        blocks.append(f"  Error:                      {error_type}: {state.get('error_message')}")
    blocks.extend(
        [
            "",
            f"Источники ({len(state['sources'])})",
            thin,
        ]
    )

    for src in state["sources"]:
        ref = src.get("doc_ref", "?")
        fn = src.get("filename", "")
        dt = src.get("document_title") or ""
        st = src.get("section_title") or ""
        blocks.append(f"  {ref} {fn}")
        if dt:
            blocks.append(f"    document: {dt}")
        if st:
            blocks.append(f"    section:  {st}")
        blocks.append("")

    blocks.extend(
        [
            f"Документы ({len(ranked_docs)})",
            thin,
        ]
    )
    for i, rd in enumerate(ranked_docs, 1):
        doc = rd["doc"]
        name = doc.metadata.get("filename", "unknown")
        preview = (doc.page_content or "").replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:220] + "…"

        ret = rd.get("retrieval_score")
        rer = rd.get("rerank_score")
        final = rd.get("final_score")
        ret_str = f"{ret:.4f}" if isinstance(ret, (int, float)) else "n/a"
        rer_str = f"{rer:.4f}" if isinstance(rer, (int, float)) else "n/a"
        final_str = f"{final:.4f}" if isinstance(final, (int, float)) else "n/a"

        blocks.append(f"  {i}. {name}")
        blocks.append(f"     score: {final_str} (retrieval={ret_str}, rerank={rer_str})")
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


# Runnable wrapper allows piping graph output through LangChain-style chains if needed.
render_rag_output = RunnableLambda(_format_rag_state)


def main() -> None:
    """Run a REPL: read Russian questions, print RAG state and answer, until EOF."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    rag_pipeline, reranker = build_rag_stack()
    try:
        while True:
            query = input("Введите Ваш вопрос: ")
            result: RAGState = rag_pipeline.run(query)
            sys.stdout.write(render_rag_output.invoke(result))
            sys.stdout.write("\n")
    finally:
        reranker.close()


if __name__ == "__main__":
    main()
