import logging
from pathlib import Path

from qa_dataset_generator.generator import QADatasetGenerator

logger = logging.getLogger(__name__)


def generate_qa_dataset(
    sections_file: Path,
    output_file: Path,
    model: str,
    base_url: str | None,
    load_api_key: bool,
    temperature: float,
    max_context: int,
    sections_per_pdf: int,
    seed: int | None,
) -> None:
    generator = QADatasetGenerator(
        model_name=model,
        base_url=base_url,
        load_api_key=load_api_key,
        temperature=temperature,
        max_context_length=max_context,
        sections_per_pdf=sections_per_pdf,
        seed=seed,
    )

    pairs = generator.generate_from_sections_file(
        sections_file=sections_file,
        output_file=output_file,
    )

    stats_by_type: dict[str, int] = {}
    stats_by_section: dict[str, int] = {}

    for pair in pairs:
        stats_by_type[pair.question_type] = stats_by_type.get(pair.question_type, 0) + 1
        stats_by_section[pair.section_type] = stats_by_section.get(pair.section_type, 0) + 1

    logger.info("=" * 80)
    logger.info("GENERATION STATISTICS")
    logger.info("=" * 80)
    logger.info("Total QA pairs: %s", len(pairs))
    logger.info("By question type:")
    for qtype, count in sorted(stats_by_type.items()):
        logger.info("  - %s: %s", qtype, count)

    logger.info("By section type:")
    for stype, count in sorted(stats_by_section.items()):
        logger.info("  - %s: %s", stype, count)
