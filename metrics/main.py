import argparse
from pathlib import Path

from cadence_md.app.embedder import get_embedder
from cadence_md.app.llm import get_llm
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.rag import RAGPipeline
from cadence_md.app.reranker import get_reranker
from cadence_md.app.settings import settings
from metrics.config import metrics_settings
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


def build_evaluation_pipeline() -> RAGEvaluationPipeline:
    """Create and initialize evaluation dependencies lazily."""
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


def build_metrics_arg_parser() -> argparse.ArgumentParser:
    """CLI for metrics evaluation (full RAG vs retriever-only)."""
    parser = argparse.ArgumentParser(description="RAG evaluation (metrics/)")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Full evaluation pipeline (retrieval + rerank + generator)
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

    # Retriever-only evaluation (no RAGAS / no answer generation)
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
    return parser


def run_metrics_cli(argv: list[str] | None = None) -> None:
    """Parse argv and run the selected evaluation mode."""
    args = build_metrics_arg_parser().parse_args(argv)
    evaluation_pipeline = build_evaluation_pipeline()
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
