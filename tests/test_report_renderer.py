"""Tests for Markdown validation report rendering."""

import pandas as pd

from metrics.report_renderer import (
    _render_artifacts_section,
    _render_breakdown_sections,
    _render_problem_cases_section,
    _render_ragas_metrics_section,
    _render_text_match_retriever_metrics_section,
    render_validation_report,
)


def test_render_text_match_section_empty_when_missing() -> None:
    lines = _render_text_match_retriever_metrics_section(
        summary_metrics={"retrieval_metrics": {"k": 5}}
    )
    assert lines == []


def test_render_ragas_metrics_section_empty_summary() -> None:
    lines = _render_ragas_metrics_section(summary_metrics={"ragas_metrics": {}})
    text = "".join(lines)
    assert "## RAGAS Metrics" in text
    assert "No successful RAGAS metric rows" in text


def test_render_breakdown_sections_none_or_empty() -> None:
    assert _render_breakdown_sections(ragas_df=None) == []
    assert _render_breakdown_sections(ragas_df=pd.DataFrame()) == []


def test_render_breakdown_sections_counts_and_nan() -> None:
    df = pd.DataFrame(
        {
            "question_type": ["a", "a", None],
            "section_type": ["x", "y", "x"],
        }
    )
    lines = _render_breakdown_sections(ragas_df=df)
    text = "".join(lines)
    assert "`a` (n=2)" in text
    assert "`x` (n=2)" in text or "`y` (n=1)" in text


def test_render_problem_cases_no_preview_when_empty() -> None:
    lines = _render_problem_cases_section(missed_retrieval_case_ids=[])
    text = "".join(lines)
    assert "Missed retrieval cases: 0" in text
    assert "preview" not in text.lower()


def test_render_problem_cases_preview_truncated_to_20() -> None:
    ids = list(range(30))
    lines = _render_problem_cases_section(missed_retrieval_case_ids=ids)
    text = "".join(lines)
    assert "Missed retrieval cases: 30" in text
    assert "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19" in text
    assert ", 20," not in text


def test_render_artifacts_section_full_vs_retriever() -> None:
    full_text = "".join(_render_artifacts_section(mode="full"))
    assert "`cases.jsonl`" in full_text
    assert "`ragas_scores.parquet`" in full_text
    assert "`retrieval_cases.jsonl`" not in full_text

    ret_text = "".join(_render_artifacts_section(mode="retriever"))
    assert "`retrieval_cases.jsonl`" in ret_text
    assert "`cases.jsonl`" not in ret_text


def test_render_validation_report_retriever_skips_ragas_sections() -> None:
    manifest = {
        "run_id": "r1",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "paths": {"dataset_file": "d.jsonl"},
        "run_parameters": {"sample_size": None, "k": 3},
        "rag_config": {
            "llm_model": "m",
            "embedding_model": "e",
            "reranker_model": "r",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "retrieval": {
                "search_mode": "dense",
                "fusion_method": "none",
                "sparse_top_k": 1,
                "dense_top_k": 5,
                "hybrid_top_k": 5,
            },
            "qdrant_collection": "c",
        },
    }
    summary = {
        "total_loaded_cases": 1,
        "evaluated_cases": 1,
        "rag_success_cases": 1,
        "rag_errors": 0,
        "ragas_errors": 0,
        "retrieval_metrics": {
            "k": 3,
            "hit_rate": 0.5,
            "mrr": 0.25,
            "recall_at_k": 0.33,
            "precision_at_k": 0.11,
            "avg_score": 0.88,
        },
    }
    text = render_validation_report(
        manifest=manifest,
        summary_metrics=summary,
        ragas_df=pd.DataFrame([{"faithfulness": 0.9}]),
        mode="retriever",
        missed_retrieval_case_ids=[],
    )
    assert "## RAGAS Metrics" not in text
    assert "Breakdown by Question Type" not in text
    assert "## Retriever Metrics" in text
    assert "0.500" in text or "0.5" in text


def test_render_validation_report_full_includes_ragas_and_breakdown() -> None:
    manifest = {
        "run_id": "f1",
        "timestamp_utc": "2026-01-02T00:00:00+00:00",
        "paths": {"dataset_file": "q.jsonl"},
        "run_parameters": {"sample_size": 5, "k": 5},
        "rag_config": {
            "llm_model": "m",
            "embedding_model": "e",
            "reranker_model": "r",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "retrieval": {
                "search_mode": "dense",
                "fusion_method": "none",
                "sparse_top_k": 1,
                "dense_top_k": 5,
                "hybrid_top_k": 5,
            },
            "qdrant_collection": "c",
        },
    }
    summary = {
        "total_loaded_cases": 2,
        "evaluated_cases": 2,
        "rag_success_cases": 2,
        "rag_errors": 0,
        "ragas_errors": 0,
        "retrieval_metrics": {
            "k": 5,
            "hit_rate": 1.0,
            "mrr": 1.0,
            "recall_at_k": 1.0,
            "precision_at_k": 0.2,
            "avg_score": 0.91,
        },
        "ragas_metrics": {
            "faithfulness": {
                "count": 2,
                "mean": 0.85,
                "median": 0.85,
                "std": 0.0,
                "min": 0.85,
                "max": 0.85,
            }
        },
    }
    ragas_df = pd.DataFrame(
        [
            {
                "faithfulness": 0.8,
                "question_type": "factoid",
                "section_type": "therapy",
            },
            {
                "faithfulness": 0.9,
                "question_type": "procedural",
                "section_type": "therapy",
            },
        ]
    )
    text = render_validation_report(
        manifest=manifest,
        summary_metrics=summary,
        ragas_df=ragas_df,
        mode="full",
        missed_retrieval_case_ids=[0],
    )
    assert "## RAGAS Metrics" in text
    assert "### faithfulness" in text
    assert "0.850" in text
    assert "Breakdown by Question Type" in text
    assert "`factoid` (n=1)" in text
    assert "## Text Matcher Retriever Metrics" not in text
