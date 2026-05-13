"""Human-readable Markdown report for QA dataset generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from qa_dataset_generator.schemas import GenerationPipelineStats


def _markdown_table_two_columns(headers: tuple[str, str], rows: list[tuple[str, str]]) -> str:
    """
    Build a markdown table with two columns

    Args:
        headers: Tuple of header strings
        rows: List of tuples of row values
    Returns:
        String containing the markdown table
    """
    h0, h1 = headers
    lines = [
        f"| {h0} | {h1} |",
        "| --- | --- |",
    ]
    for a, b in rows:
        lines.append(f"| {a} | {b} |")
    return "\n".join(lines)


def build_generation_report_markdown(
    *,
    sections_file: Path,
    output_file: Path,
    model: str,
    temperature: float,
    max_context: int,
    min_context: int,
    sections_per_pdf: int,
    seed: int | None,
    pipeline_stats: GenerationPipelineStats,
    stats_by_question_type: dict[str, int],
    stats_by_section_type: dict[str, int],
    generated_at_utc: datetime,
) -> str:
    """
    Build report text without writing to disk

    Args:
        sections_file: Path to the sections file
        output_file: Path to the output file
        model: Name of the model
        temperature: Temperature for the model
        max_context: Maximum context length
        min_context: Minimum context length
        sections_per_pdf: Number of sections per PDF
        seed: Seed for the random number generator
        pipeline_stats: Statistics for the generation pipeline
        stats_by_question_type: Statistics by question type
        stats_by_section_type: Statistics by section type
        generated_at_utc: Timestamp of generation
    Returns:
        String containing the report text
    """
    # Initialize variables
    seed_display = str(seed) if seed is not None else "—"
    ts = generated_at_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build run parameters table
    run_params_rows = [
        ("Sections file", f"`{sections_file}`"),
        ("Output JSONL", f"`{output_file}`"),
        ("Model", model),
        ("Temperature", str(temperature)),
        ("Max context length (characters)", str(max_context)),
        ("Min context length (characters)", str(min_context)),
        ("Sections per PDF (sample cap)", str(sections_per_pdf)),
        ("Seed", seed_display),
    ]

    # Build pipeline summary table
    pipeline_rows = [
        ("Sections loaded from JSONL", str(pipeline_stats.sections_loaded)),
        ("Sections selected for generation", str(pipeline_stats.sections_selected)),
        ("Skipped invalid JSONL lines", str(pipeline_stats.skipped_invalid_jsonl)),
        ("Sources total", str(pipeline_stats.sources_total)),
        ("Sources skipped (no long sections)", str(pipeline_stats.sources_skipped_short)),
        ("Sections filtered (too short)", str(pipeline_stats.sections_filtered_short)),
        ("Sections processed (iterations)", str(pipeline_stats.sections_processed)),
        ("QA pairs generated successfully", str(pipeline_stats.pairs_generated)),
        ("Failed sections (error or empty response)", str(pipeline_stats.sections_failed)),
    ]

    # Build distribution by question type table
    q_rows = [(k, str(v)) for k, v in sorted(stats_by_question_type.items())]

    # Build distribution by section type table
    s_rows = [(k, str(v)) for k, v in sorted(stats_by_section_type.items())]

    # Build report parts
    parts = [
        "# QA dataset generation report",
        "",
        f"**Generated at:** {ts}",
        "",
        "## Run parameters",
        "",
        _markdown_table_two_columns(("Parameter", "Value"), run_params_rows),
        "",
        "## Pipeline summary",
        "",
        _markdown_table_two_columns(("Metric", "Value"), pipeline_rows),
        "",
        "## Skipped sources",
        "",
    ]
    if pipeline_stats.skipped_sources:
        parts.extend(f"- `{source}`" for source in pipeline_stats.skipped_sources)
    else:
        parts.append("_None._")
    parts.extend(
        [
            "",
            "## Distribution by question type",
            "",
        ]
    )
    if q_rows:
        parts.append(_markdown_table_two_columns(("Question type", "Count"), q_rows))
    else:
        parts.append("_No successfully generated pairs._")
    parts.extend(
        [
            "",
            "## Distribution by section type",
            "",
        ]
    )
    if s_rows:
        parts.append(_markdown_table_two_columns(("Section type", "Count"), s_rows))
    else:
        parts.append("_No successfully generated pairs._")
    parts.append("")
    return "\n".join(parts)
