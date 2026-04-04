import argparse
from pathlib import Path


def _run_metrics_cli_argv(argv: list[str]) -> None:
    """Lazy import so project CLI does not load RAG stack unless needed."""
    from metrics.main import run_metrics_cli  # noqa: PLC0415

    run_metrics_cli(argv)


def main():
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
        action="store_true",
        default=True,
        help="Load api key from environment if True",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM answer generation (0.0-1.0)",
    )
    gen_parser.add_argument(
        "--max-context",
        type=int,
        default=10000,
        help="Maximum context length in characters",
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
        type=int,
        default=None,
        help="Optional cap on number of test cases",
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
        type=int,
        default=None,
        help="Optional cap on number of test cases",
    )
    metrics_ret.add_argument(
        "--k",
        type=int,
        default=None,
        help="K for recall@K / precision@K",
    )

    args = parser.parse_args()

    if args.command == "generate-qa":
        from qa_dataset_generator.generate_qa_dataset import (  # noqa: PLC0415
            generate_qa_dataset,
        )

        generate_qa_dataset(
            sections_file=args.sections_file,
            output_file=args.output_file,
            model=args.model,
            base_url=args.base_url,
            load_api_key=args.load_api_key,
            temperature=args.temperature,
            max_context=args.max_context,
        )
    elif args.command == "metrics-eval-full":
        argv: list[str] = [
            "full",
            "--dataset-file",
            str(args.dataset_file),
            "--output-dir",
            str(args.output_dir),
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
        ]
        if args.sample_size is not None:
            argv_ret += ["--sample-size", str(args.sample_size)]
        if args.k is not None:
            argv_ret += ["--k", str(args.k)]
        _run_metrics_cli_argv(argv_ret)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
