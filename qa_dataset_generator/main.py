import logging
from datetime import UTC, datetime
from pathlib import Path

from qa_dataset_generator.config import generation_report_path
from qa_dataset_generator.generator import QADatasetGenerator
from qa_dataset_generator.report import build_generation_report_markdown

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_qa_dataset(
    sections_file: Path,
    output_file: Path,
    model: str,
    temperature: float,
    max_context: int,
    sections_per_pdf: int,
    seed: int | None,
) -> None:
    """
    Generate a QA dataset from a sections file

    Args:
        sections_file: Path to the sections file
        output_file: Path to the output file
        model: Name of the model
        temperature: Temperature for the model
        max_context: Maximum context length
        sections_per_pdf: Number of sections per PDF
        seed: Seed for the random number generator
    """
    # Initialize generator
    generator = QADatasetGenerator(
        model_name=model,
        temperature=temperature,
        max_context_length=max_context,
        sections_per_pdf=sections_per_pdf,
        seed=seed,
    )

    # Generate QA pairs
    pairs, pipeline_stats = generator.generate_from_sections_file(
        sections_file=sections_file,
        output_file=output_file,
    )

    # Build statistics
    stats_by_type: dict[str, int] = {}
    stats_by_section: dict[str, int] = {}
    for pair in pairs:
        stats_by_type[pair.question_type] = stats_by_type.get(pair.question_type, 0) + 1
        stats_by_section[pair.section_type] = stats_by_section.get(pair.section_type, 0) + 1

    generated_at = datetime.now(UTC)

    # Build report
    report_body = build_generation_report_markdown(
        sections_file=sections_file,
        output_file=output_file,
        model=model,
        temperature=temperature,
        max_context=max_context,
        min_context=pipeline_stats.min_context_length,
        sections_per_pdf=sections_per_pdf,
        seed=seed,
        pipeline_stats=pipeline_stats,
        stats_by_question_type=stats_by_type,
        stats_by_section_type=stats_by_section,
        generated_at_utc=generated_at,
    )

    # Write report
    report_path = generation_report_path(output_file)
    report_path.write_text(report_body, encoding="utf-8")
    logger.info(f"Generation report written to {report_path}")
    logger.info("Generation complete")
