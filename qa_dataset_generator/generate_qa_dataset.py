from pathlib import Path

from qa_dataset_generator.generator import QADatasetGenerator


def generate_qa_dataset(
    sections_file: Path,
    output_file: Path,
    model: str,
    base_url: str | None,
    load_api_key: bool,
    temperature: float,
    max_context: int,
):
    generator = QADatasetGenerator(
        model_name=model,
        base_url=base_url,
        load_api_key=load_api_key,
        temperature=temperature,
        max_context_length=max_context,
    )

    pairs = generator.generate_from_sections_file(
        sections_file=sections_file,
        output_file=output_file,
    )

    print("\n" + "=" * 80)
    print("GENERATION STATISTICS")
    print("=" * 80)

    stats_by_type: dict[str, int] = {}
    stats_by_section: dict[str, int] = {}

    for pair in pairs:
        stats_by_type[pair.question_type] = stats_by_type.get(pair.question_type, 0) + 1
        stats_by_section[pair.section_type] = stats_by_section.get(pair.section_type, 0) + 1

    print(f"\nTotal QA pairs: {len(pairs)}")
    print("\nBy question type:")
    for qtype, count in sorted(stats_by_type.items()):
        print(f"  - {qtype}: {count}")

    print("\nBy section type:")
    for stype, count in sorted(stats_by_section.items()):
        print(f"  - {stype}: {count}")
