"""Tests for config helpers, schemas serialization, and report table helpers."""

import json
from datetime import UTC, datetime
from pathlib import Path

from qa_dataset_generator.config import generation_report_path
from qa_dataset_generator.report import (
    _markdown_table_two_columns,
    build_generation_report_markdown,
)
from qa_dataset_generator.schemas import GenerationPipelineStats, QAPair, QAResponsePair


def test_generation_report_path_next_to_jsonl() -> None:
    assert generation_report_path(Path("out/foo.jsonl")) == Path("out/foo_generation_report.md")


def test_markdown_table_two_columns_empty_rows() -> None:
    text = _markdown_table_two_columns(("A", "B"), [])
    assert "| A | B |" in text
    assert "| --- | --- |" in text
    assert text.splitlines() == ["| A | B |", "| --- | --- |"]


def test_build_generation_report_markdown_seed_none_displays_em_dash() -> None:
    pipeline = GenerationPipelineStats(
        sections_loaded=1,
        sections_selected=1,
        skipped_invalid_jsonl=0,
        sections_processed=1,
        pairs_generated=1,
        sections_failed=0,
        min_context_length=500,
        sources_total=1,
        sources_skipped_short=0,
        skipped_sources=(),
        sections_filtered_short=0,
    )
    text = build_generation_report_markdown(
        sections_file=Path("s.jsonl"),
        output_file=Path("o.jsonl"),
        model="m",
        temperature=0.0,
        max_context=100,
        min_context=50,
        sections_per_pdf=3,
        seed=None,
        pipeline_stats=pipeline,
        stats_by_question_type={"simple": 1},
        stats_by_section_type={"treatment": 1},
        generated_at_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
    )
    assert "| Seed | — |" in text


def test_qa_pair_model_dump_json_roundtrip_fields() -> None:
    pair = QAPair(
        question="Вопрос достаточной длины?",
        answer="Ответ достаточной длины для теста.",
        question_type="simple",
        section_type="treatment",
        section_title="Заголовок",
        document_title="Документ",
        mkb_codes=["A00"],
        context="Контекст",
        section_id="section_abc",
    )
    raw = pair.model_dump_json()
    data = json.loads(raw)
    assert data["section_id"] == "section_abc"
    assert data["question_type"] == "simple"


def test_qa_response_pair_model_validate() -> None:
    r = QAResponsePair.model_validate(
        {
            "question": "клинический вопрос",
            "answer": "краткий ответ",
            "context": "цитата из фрагмента",
        }
    )
    assert r.question == "клинический вопрос"
    assert "цитата" in r.context
