from pathlib import Path

from cadence_md.app.embedder import get_embedder
from cadence_md.app.llm import get_llm
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag import RAGPipeline
from cadence_md.app.reranker import get_reranker
from cadence_md.app.settings import EmbeddingConfig, QdrantConfig, RAGConfig

# Init configuration
qdrant_config = QdrantConfig(rebuild_collection=False)
embedding_config = EmbeddingConfig(return_score=True)
baseline_config = RAGConfig(embedding=embedding_config, qdrant_config=qdrant_config)

# Init models connection
embedder = get_embedder(baseline_config)
reranker = get_reranker(baseline_config)
llm = get_llm(baseline_config)

# Init qdrant
qdrant_manager = QdrantManager(config=baseline_config, embedder=embedder)
qdrant_manager.setup_qdrant(data_dir=Path("../data/clinical_recomendation_pdfs/"))

# Init RAG graph
rag_pipeline = RAGPipeline(llm, qdrant_manager, reranker=reranker)


def main():
    # Run RAG
    while True:
        query = input("Введите Ваш вопрос:")
        result = rag_pipeline.run(query)

        print(f"QUERY: {result['query']}")
        print()
        print(f"\nRETRIEVED DOCUMENTS ({len(result['retrieved_docs'])}):")
        for i, doc in enumerate(result["retrieved_docs"], 1):
            print(f"\n{i}. {doc.metadata['filename']} (score: {doc.metadata['score']:.4f})")
            print(f"   {doc.page_content[:200]}...")
            print()

        print("ANSWER:")
        print(result["answer"])
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
