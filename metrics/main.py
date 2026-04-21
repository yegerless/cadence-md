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

DEFAULT_PDF_DIR = Path("data/main_specialities/")


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _existing_dir(value: str) -> Path:
    """Parse an existing directory path for CLI."""
    directory = Path(value)
    if not directory.exists():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise argparse.ArgumentTypeError(f"Expected directory path, got file: {directory}")
    return directory


def build_evaluation_pipeline(pdf_dir: Path) -> RAGEvaluationPipeline:
    """Create and initialize evaluation dependencies lazily."""
    embedder = get_embedder()
    reranker = get_reranker()
    llm = get_llm()

    qdrant_manager = QdrantManager(embedder)
    qdrant_manager.setup_qdrant(data_dir=pdf_dir)
    rag_pipeline = RAGPipeline(llm, qdrant_manager, reranker=reranker)

    gigachat_api_key = os.getenv("GIGACHAT_API_KEY")
    gigachat_llm = ThrottledGigaChat(
        credentials=gigachat_api_key,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        temperature=0.0,
    )
    gigachat_embeddings = ThrottledGigaChatEmbeddings(
        credentials=gigachat_api_key,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="Embeddings",
    )

    return RAGEvaluationPipeline(
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
        type=_positive_int,
        default=None,
        help="Optional cap on number of test cases",
    )
    full_p.add_argument(
        "--k",
        type=_positive_int,
        default=5,
        help="K for recall@K / precision@K (default: 5)",
    )
    full_p.add_argument(
        "--pdf-dir",
        type=_existing_dir,
        default=DEFAULT_PDF_DIR,
        help="Directory with source PDF files used by Qdrant setup",
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
        type=_positive_int,
        default=None,
        help="Optional cap on number of test cases",
    )
    ret_p.add_argument(
        "--k",
        type=_positive_int,
        default=None,
        help="K for recall@K / precision@K (default: pipeline default)",
    )
    ret_p.add_argument(
        "--pdf-dir",
        type=_existing_dir,
        default=DEFAULT_PDF_DIR,
        help="Directory with source PDF files used by Qdrant setup",
    )
    return parser


def run_metrics_cli(argv: list[str] | None = None) -> None:
    """Parse argv and run the selected evaluation mode."""
    args = build_metrics_arg_parser().parse_args(argv)
    evaluation_pipeline = build_evaluation_pipeline(args.pdf_dir)
    if args.mode == "full":
        evaluation_pipeline.run_full_evaluation(
            dataset_file=args.dataset_file,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            k=args.k,
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
