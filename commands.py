import argparse
from pathlib import Path

from qa_dataset_generator.generate_qa_dataset import generate_qa_dataset


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
        default="GigaChat",
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
        default=50000,
        help="Maximum context length in characters",
    )

    # other_parser = subparsers.add_parser("something", help="...")

    args = parser.parse_args()

    if args.command == "generate-qa":
        generate_qa_dataset(
            sections_file=args.sections_file,
            output_file=args.output_file,
            model=args.model,
            base_url=args.base_url,
            load_api_key=args.load_api_key,
            temperature=args.temperature,
            max_context=args.max_context,
        )
    # elif args.command == "something":
    #     other_command_fn(...)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
