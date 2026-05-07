import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from qa_dataset_generator.config import (
    DEFAULT_MAX_CONTEXT_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_SECTIONS_PER_PDF,
    DEFAULT_TEMPERATURE,
)

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """Parse positive integer from CLI."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _add_rag_optional_node_flags(
    parser: argparse.ArgumentParser, *, include_formatter: bool
) -> None:
    """Add optional RAG graph override flags to a metrics subcommand."""
    parser.add_argument(
        "--disable-rag-query-rewriter",
        action="store_true",
        help="Disable optional RAG query rewriting for this metrics run",
    )
    parser.add_argument(
        "--disable-rag-context-relevance-grader",
        action="store_true",
        help="Disable optional RAG context relevance grading for this metrics run",
    )
    if include_formatter:
        parser.add_argument(
            "--disable-rag-answer-formatter",
            action="store_true",
            help="Disable optional RAG answer formatting for this full metrics run",
        )


def _rag_optional_node_overrides(args: argparse.Namespace) -> dict[str, bool]:
    """Translate optional CLI disable flags into partial RAGOptionalNodesConfig updates."""
    overrides: dict[str, bool] = {}
    if getattr(args, "disable_rag_query_rewriter", False):
        overrides["enable_query_rewriter"] = False
    if getattr(args, "disable_rag_context_relevance_grader", False):
        overrides["enable_context_relevance_grader"] = False
    if getattr(args, "disable_rag_answer_formatter", False):
        overrides["enable_answer_formatter"] = False
    return overrides


def build_project_cli_parser() -> argparse.ArgumentParser:
    """Argument parser for the project CLI (tests and programmatic use)."""
    parser = argparse.ArgumentParser(
        description="Project CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # command generate-qa
    gen_parser = subparsers.add_parser(
        "generate-qa",
        help="Generate a synthetic QA dataset from clinical_sections.jsonl",
    )
    gen_parser.add_argument(
        "--sections-file",
        type=Path,
        default=Path("data/clinical_sections.jsonl"),
        help="Path to the JSONL file with sections (parser result)",
    )
    gen_parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/metrics_evaluation_datasets/qa_dataset.jsonl"),
        help="Path to output JSONL file",
    )
    gen_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="LLM model name (GigaChat, GigaChat-2-Max, GigaChat-pro)",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for LLM answer generation (0.0-1.0)",
    )
    gen_parser.add_argument(
        "--max-context",
        type=_positive_int,
        default=DEFAULT_MAX_CONTEXT_LENGTH,
        help="Maximum context length in characters",
    )
    gen_parser.add_argument(
        "--sections-per-pdf",
        type=_positive_int,
        default=DEFAULT_SECTIONS_PER_PDF,
        help="Randomly sample up to N sections from each source PDF",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )

    # command metrics-eval-full
    metrics_full = subparsers.add_parser(
        "metrics-eval-full",
        help="Full RAG evaluation (RAGAS + retrieval metrics + report)",
    )
    metrics_full.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("data/metrics_evaluation_datasets/qa_dataset.jsonl"),
        help="Path to QA JSONL",
    )
    metrics_full.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics/results/"),
        help="Directory for reports and metrics",
    )
    metrics_full.add_argument(
        "--sample-size",
        type=_positive_int,
        default=None,
        help="Optional cap on number of test cases",
    )
    metrics_full.add_argument(
        "--k",
        type=_positive_int,
        default=5,
        help="K for recall@K / precision@K",
    )
    metrics_full.add_argument(
        "--enable-text-matcher-metrics",
        action="store_true",
        help="Also compute prefixed retrieval metrics using text matcher",
    )
    _add_rag_optional_node_flags(metrics_full, include_formatter=True)

    # command metrics-eval-retriever
    metrics_ret = subparsers.add_parser(
        "metrics-eval-retriever",
        help="Retriever-only evaluation (no RAGAS / no answer generation)",
    )
    metrics_ret.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("data/metrics_evaluation_datasets/qa_dataset.jsonl"),
        help="Path to QA JSONL",
    )
    metrics_ret.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics/results/"),
        help="Directory for reports and metrics",
    )
    metrics_ret.add_argument(
        "--sample-size",
        type=_positive_int,
        default=None,
        help="Optional cap on number of test cases",
    )
    metrics_ret.add_argument(
        "--k",
        type=_positive_int,
        default=None,
        help="K for recall@K / precision@K (default: pipeline default)",
    )
    metrics_ret.add_argument(
        "--enable-text-matcher-metrics",
        action="store_true",
        help="Also compute prefixed retrieval metrics using text matcher",
    )
    metrics_ret.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Parallel worker threads for retriever cases",
    )
    _add_rag_optional_node_flags(metrics_ret, include_formatter=False)

    # command parse-pdf
    parse_pdf = subparsers.add_parser(
        "parse-pdf",
        help="Parse clinical guideline PDFs into sections JSONL",
    )
    parse_pdf.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory with source PDF files",
    )
    parse_pdf.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSONL file for parsed sections",
    )
    parse_pdf.add_argument(
        "--max-files",
        type=_positive_int,
        default=None,
        help="Optional cap on number of PDF files to parse",
    )

    return parser


def _run_parse_pdf(
    pdf_dir: Path,
    output_file: Path,
    max_files: int | None,
) -> None:
    """Parse clinical guideline PDFs and store sections JSONL."""
    from cadence_md.app.pdf_parser import ClinicalGuidelinesParser  # noqa: PLC0415

    parser = ClinicalGuidelinesParser()
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if max_files is not None:
        pdf_files = pdf_files[:max_files]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    sections = []
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs", unit="file"):
        parsed = parser.parse_pdf(pdf_file)
        if parsed:
            sections.extend(parsed)

    with output_file.open("w", encoding="utf-8") as f:
        for section in sections:
            f.write(json.dumps(section.to_dict(), ensure_ascii=False) + "\n")

    logger.info(
        "Parsed %s files, saved %s sections to %s", len(pdf_files), len(sections), output_file
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_project_cli_parser()
    args = parser.parse_args()

    if args.command == "generate-qa":
        from qa_dataset_generator.main import (  # noqa: PLC0415
            generate_qa_dataset,
        )

        try:
            generate_qa_dataset(
                sections_file=args.sections_file,
                output_file=args.output_file,
                model=args.model,
                temperature=args.temperature,
                max_context=args.max_context,
                sections_per_pdf=args.sections_per_pdf,
                seed=args.seed,
            )
        except FileExistsError as error:
            logger.error(
                "Output file already exists: %s. Choose a different --output-file path.",
                error,
            )
            raise SystemExit(2) from error
    elif args.command in ("metrics-eval-full", "metrics-eval-retriever"):
        from metrics.main import build_evaluation_pipeline  # noqa: PLC0415

        evaluation_pipeline = build_evaluation_pipeline(
            optional_nodes_overrides=_rag_optional_node_overrides(args)
        )
        if args.command == "metrics-eval-full":
            evaluation_pipeline.run_full_evaluation(
                dataset_file=args.dataset_file,
                output_dir=args.output_dir,
                sample_size=args.sample_size,
                k=args.k,
                enable_text_matcher_metrics=args.enable_text_matcher_metrics,
            )
        else:
            evaluation_pipeline.run_retriever_evaluation(
                dataset_file=args.dataset_file,
                output_dir=args.output_dir,
                sample_size=args.sample_size,
                k=args.k,
                enable_text_matcher_metrics=args.enable_text_matcher_metrics,
                workers=args.workers,
            )
    elif args.command == "parse-pdf":
        _run_parse_pdf(
            pdf_dir=args.pdf_dir,
            output_file=args.output_file,
            max_files=args.max_files,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
