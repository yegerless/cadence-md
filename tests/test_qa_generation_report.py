"""Tests for QA generation Markdown report (no LLM)."""

from datetime import UTC, datetime
from pathlib import Path

from qa_dataset_generator.report import build_generation_report_markdown
from qa_dataset_generator.schemas import GenerationPipelineStats


def test_build_generation_report_markdown_contains_expected_sections() -> None:
    pipeline = GenerationPipelineStats(
        sections_loaded=100,
        sections_selected=10,
        skipped_invalid_jsonl=2,
        sections_processed=10,
        pairs_generated=8,
        sections_failed=2,
        min_context_length=500,
        sources_total=12,
        sources_skipped_short=2,
        skipped_sources=("short_doc_a.pdf", "short_doc_b.pdf"),
        sections_filtered_short=15,
    )
    text = build_generation_report_markdown(
        sections_file=Path("data/in/sections.jsonl"),
        output_file=Path("data/out/qa_dataset.jsonl"),
        model="GigaChat-2-Max",
        temperature=0.0,
        max_context=10000,
        min_context=500,
        sections_per_pdf=3,
        seed=42,
        pipeline_stats=pipeline,
        stats_by_question_type={"simple": 5, "reasoning": 3},
        stats_by_section_type={"diagnosis": 4, "treatment": 4},
        generated_at_utc=datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC),
    )
    assert "# QA dataset generation report" in text
    assert "2026-05-02 12:00:00 UTC" in text
    assert "`data/in/sections.jsonl`" in text
    assert "`data/out/qa_dataset.jsonl`" in text
    assert "GigaChat-2-Max" in text
    assert "| Sections loaded from JSONL | 100 |" in text
    assert "| QA pairs generated successfully | 8 |" in text
    assert "| reasoning | 3 |" in text
    assert "| simple | 5 |" in text
    assert "| diagnosis | 4 |" in text
    assert "| Min context length (characters) | 500 |" in text
    assert "| Sources total | 12 |" in text
    assert "| Sources skipped (no long sections) | 2 |" in text
    assert "| Sections filtered (too short) | 15 |" in text
    assert "## Skipped sources" in text
    assert "- `short_doc_a.pdf`" in text
    assert "- `short_doc_b.pdf`" in text


def test_build_generation_report_markdown_no_skipped_sources() -> None:
    pipeline = GenerationPipelineStats(
        sections_loaded=10,
        sections_selected=5,
        skipped_invalid_jsonl=0,
        sections_processed=5,
        pairs_generated=5,
        sections_failed=0,
        min_context_length=500,
        sources_total=2,
        sources_skipped_short=0,
        skipped_sources=(),
        sections_filtered_short=0,
    )
    text = build_generation_report_markdown(
        sections_file=Path("data/in/sections.jsonl"),
        output_file=Path("data/out/qa_dataset.jsonl"),
        model="GigaChat-2-Max",
        temperature=0.0,
        max_context=10000,
        min_context=500,
        sections_per_pdf=3,
        seed=42,
        pipeline_stats=pipeline,
        stats_by_question_type={"simple": 5},
        stats_by_section_type={"diagnosis": 5},
        generated_at_utc=datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC),
    )
    assert "## Skipped sources" in text
    assert "_None._" in text
