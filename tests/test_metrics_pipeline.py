import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document

from metrics.evaluation_pipeline import RAGEvaluationPipeline
from metrics.main import build_metrics_arg_parser
from metrics.schemas import RAGTestResult


def _pipeline_without_init() -> RAGEvaluationPipeline:
    pipeline = object.__new__(RAGEvaluationPipeline)
    pipeline.ragas_metrics = []
    return pipeline


def test_load_test_cases_skips_malformed_rows(tmp_path: Path) -> None:
    dataset_file = tmp_path / "qa_dataset.jsonl"
    valid_row = {
        "question": "Q1",
        "answer": "A1",
        "context": "C1",
        "question_type": "factoid",
        "section_type": "diagnostics",
    }
    missing_required = {
        "question": "Q2",
        "answer": "A2",
    }
    dataset_file.write_text(
        "\n".join(
            [
                json.dumps(valid_row, ensure_ascii=False),
                '{"broken_json":',
                json.dumps(missing_required, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pipeline = _pipeline_without_init()
    cases = pipeline.load_test_cases(dataset_file)

    assert len(cases) == 1
    assert cases[0].question == "Q1"


def test_calculate_retrieval_metrics_requires_positive_k() -> None:
    pipeline = _pipeline_without_init()
    result = RAGTestResult(
        question="Q",
        ground_truth_answer="A",
        ground_truth_context="контекст рекомендации",
        retrieved_contexts=[Document(page_content="контекст рекомендации", metadata={})],
        retrieval_scores=[0.9],
        generated_answer="",
        question_type="factoid",
        section_type="therapy",
        test_case_id=0,
    )

    with pytest.raises(ValueError, match="positive integer"):
        pipeline.calculate_retrieval_metrics([result], k=0)


def test_contexts_match_requires_meaningful_overlap() -> None:
    gt = "пациенту рекомендуется ацетилсалициловая кислота в дозе 75 мг ежедневно"
    good_retrieved = "ацетилсалициловая кислота в дозе 75 мг ежедневно назначается пациенту"
    poor_retrieved = "ацетилсалициловая кислота"

    assert RAGEvaluationPipeline.contexts_match(gt, good_retrieved)
    assert not RAGEvaluationPipeline.contexts_match(gt, poor_retrieved)


def test_generate_report_with_empty_ragas_dataframe(tmp_path: Path) -> None:
    pipeline = _pipeline_without_init()
    ragas_df = pd.DataFrame(columns=["question_type", "section_type", "test_case_id"])
    metrics = {
        "hit_rate": 0.0,
        "mrr": 0.0,
        "avg_score": 0.0,
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "k": 5,
    }

    pipeline.generate_report(ragas_df=ragas_df, retrieval_metrics=metrics, output_dir=tmp_path)

    report_file = tmp_path / "rag_evaluation_report.md"
    csv_file = tmp_path / "rag_evaluation_detailed.csv"
    assert report_file.exists()
    assert csv_file.exists()


def test_metrics_parser_rejects_non_positive_k(tmp_path: Path) -> None:
    parser = build_metrics_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "retriever",
                "--dataset-file",
                "data/metrics_evaluation_datasets/qa_dataset.jsonl",
                "--output-dir",
                str(tmp_path),
                "--pdf-dir",
                str(tmp_path),
                "--k",
                "0",
            ]
        )


def test_metrics_parser_accepts_valid_full_args(tmp_path: Path) -> None:
    parser = build_metrics_arg_parser()
    args = parser.parse_args(
        [
            "full",
            "--dataset-file",
            "data/metrics_evaluation_datasets/qa_dataset.jsonl",
            "--output-dir",
            str(tmp_path),
            "--pdf-dir",
            str(tmp_path),
            "--sample-size",
            "10",
        ]
    )
    assert args.mode == "full"
    assert args.sample_size == 10
