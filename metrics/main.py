import argparse
import os
from pathlib import Path

from cadence_md.app.embedder import get_embedder
from cadence_md.app.llm import get_llm
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag import RAGPipeline
from cadence_md.app.reranker import get_reranker
from metrics.evaluation_pipeline import RAGEvaluationPipeline
from metrics.gigachat_api_wrapper import ThrottledGigaChat, ThrottledGigaChatEmbeddings

GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")


# Init models connection
embedder = get_embedder()
reranker = get_reranker()
llm = get_llm()

# Init qdrant
qdrant_manager = QdrantManager(embedder)
qdrant_manager.setup_qdrant(data_dir=Path("data/clinical_recomendation_pdfs/"))

# Init RAG graph
rag_pipeline = RAGPipeline(llm, qdrant_manager, reranker=reranker)

gigachat_llm = ThrottledGigaChat(
    credentials=GIGACHAT_API_KEY,
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS",
    model="GigaChat-Pro",
    temperature=0.0,
)

gigachat_embeddings = ThrottledGigaChatEmbeddings(
    credentials=GIGACHAT_API_KEY,
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS",
    model="Embeddings",
)


# Init evaluation pipeline
evaluation_pipeline = RAGEvaluationPipeline(
    rag_pipeline=rag_pipeline,
    gigachat_llm=gigachat_llm,
    gigachat_embeddings=gigachat_embeddings,
)


def build_metrics_arg_parser() -> argparse.ArgumentParser:
    """CLI for metrics evaluation (full RAG vs retriever-only)."""
    parser = argparse.ArgumentParser(description="RAG evaluation (metrics/)")
    sub = parser.add_subparsers(dest="mode", required=True)

    full_p = sub.add_parser(
        "full",
        help="Full evaluation: RAGAS + retrieval metrics + report",
    )
    full_p.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("data/metrics_evaluation_datasets/qa_dataset.jsonl"),
        help="Path to QA JSONL",
    )
    full_p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics/results/"),
        help="Directory for reports and metrics",
    )
    full_p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap on number of test cases",
    )

    ret_p = sub.add_parser(
        "retriever",
        help="Retriever-only evaluation (no RAGAS / no answer generation)",
    )
    ret_p.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("data/metrics_evaluation_datasets/qa_dataset.jsonl"),
        help="Path to QA JSONL",
    )
    ret_p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics/results/"),
        help="Directory for reports and metrics",
    )
    ret_p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap on number of test cases",
    )
    ret_p.add_argument(
        "--k",
        type=int,
        default=None,
        help="K for recall@K / precision@K (default: pipeline default)",
    )
    return parser


def run_metrics_cli(argv: list[str] | None = None) -> None:
    """Parse argv and run the selected evaluation mode."""
    args = build_metrics_arg_parser().parse_args(argv)
    if args.mode == "full":
        evaluation_pipeline.run_full_evaluation(
            dataset_file=args.dataset_file,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
        )
    else:
        evaluation_pipeline.run_retriever_evaluation(
            dataset_file=args.dataset_file,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            k=args.k,
        )


def main() -> None:
    """Entry point for `python metrics/main.py ...`."""
    run_metrics_cli()


if __name__ == "__main__":
    main()
