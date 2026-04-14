import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """Parse positive integer from CLI."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _run_metrics_cli_argv(argv: list[str]) -> None:
    """Lazy import so project CLI does not load RAG stack unless needed."""
    from metrics.main import run_metrics_cli  # noqa: PLC0415

    run_metrics_cli(argv)


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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
        default="GigaChat-2-Max",
        help="LLM model name (GigaChat, GigaChat-2-Max, GigaChat-pro)",
    )
    gen_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for local model (optional)",
    )
    gen_parser.add_argument(
        "--load-api-key",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load API key from environment",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for LLM answer generation (0.0-1.0)",
    )
    gen_parser.add_argument(
        "--max-context",
        type=_positive_int,
        default=10000,
        help="Maximum context length in characters",
    )
    gen_parser.add_argument(
        "--sections-per-pdf",
        type=_positive_int,
        default=3,
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
        help="Full RAG evaluation (RAGAS + retrieval metrics), see metrics/main.py",
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
        "--pdf-dir",
        type=Path,
        default=Path("data/main_specialities/"),
        help="Directory with source PDF files used by Qdrant setup",
    )

    # command metrics-eval-retriever
    metrics_ret = subparsers.add_parser(
        "metrics-eval-retriever",
        help="Retriever-only evaluation (no RAGAS), see metrics/main.py",
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
        help="K for recall@K / precision@K",
    )
    metrics_ret.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/main_specialities/"),
        help="Directory with source PDF files used by Qdrant setup",
    )

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

    args = parser.parse_args()

    if args.command == "generate-qa":
        from qa_dataset_generator.generate_qa_dataset import (  # noqa: PLC0415
            generate_qa_dataset,
        )

        try:
            generate_qa_dataset(
                sections_file=args.sections_file,
                output_file=args.output_file,
                model=args.model,
                base_url=args.base_url,
                load_api_key=args.load_api_key,
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
    elif args.command == "metrics-eval-full":
        argv: list[str] = [
            "full",
            "--dataset-file",
            str(args.dataset_file),
            "--output-dir",
            str(args.output_dir),
            "--pdf-dir",
            str(args.pdf_dir),
        ]
        if args.sample_size is not None:
            argv += ["--sample-size", str(args.sample_size)]
        _run_metrics_cli_argv(argv)
    elif args.command == "metrics-eval-retriever":
        argv_ret: list[str] = [
            "retriever",
            "--dataset-file",
            str(args.dataset_file),
            "--output-dir",
            str(args.output_dir),
            "--pdf-dir",
            str(args.pdf_dir),
        ]
        if args.sample_size is not None:
            argv_ret += ["--sample-size", str(args.sample_size)]
        if args.k is not None:
            argv_ret += ["--k", str(args.k)]
        _run_metrics_cli_argv(argv_ret)
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
