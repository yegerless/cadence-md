import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document

from metrics.evaluation_pipeline import RAGEvaluationPipeline
from metrics.main import build_metrics_arg_parser
from metrics.schemas import QATestCase, RAGTestResult


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
            "--sample-size",
            "10",
            "--k",
            "7",
        ]
    )
    assert args.mode == "full"
    assert args.sample_size == 10
    assert args.k == 7


def test_metrics_parser_full_uses_default_k(tmp_path: Path) -> None:
    parser = build_metrics_arg_parser()
    args = parser.parse_args(
        [
            "full",
            "--dataset-file",
            "data/metrics_evaluation_datasets/qa_dataset.jsonl",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert args.k == 5


def test_run_full_evaluation_writes_run_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline_without_init()
    pipeline.rag_pipeline = type(
        "FakeRagPipeline",
        (),
        {
            "run": staticmethod(
                lambda question: {
                    "retrieved_docs": [Document(page_content=f"ctx-{question}", metadata={})],
                    "retrieved_scores": [0.9],
                    "answer": f"ans-{question}",
                }
            )
        },
    )()

    test_cases = [
        QATestCase("q1", "a1", "ctx-q1", "factoid", "therapy", "", [], {}),
        QATestCase("q2", "a2", "ctx-q2", "factoid", "therapy", "", [], {}),
    ]
    pipeline.load_test_cases = lambda _: test_cases
    pipeline.evaluate_with_ragas = lambda results: pd.DataFrame(
        [
            {
                "faithfulness": 1.0,
                "question_type": results[0].question_type,
                "section_type": results[0].section_type,
                "test_case_id": results[0].test_case_id,
            }
        ]
    )
    pipeline.calculate_retrieval_metrics = lambda *_args, **_kwargs: {
        "hit_rate": 1.0,
        "mrr": 1.0,
        "avg_score": 0.9,
        "recall_at_k": 1.0,
        "precision_at_k": 1.0,
        "k": 5,
    }
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: path.touch())

    pipeline.run_full_evaluation(
        dataset_file=Path("ignored.jsonl"),
        output_dir=tmp_path,
        sample_size=None,
        k=5,
    )

    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "cases.jsonl").exists()
    assert (run_dir / "ragas_scores.parquet").exists()
    assert (run_dir / "summary_metrics.json").exists()
    report_file = run_dir / "report.md"
    assert report_file.exists()
    report_text = report_file.read_text(encoding="utf-8")
    assert "## Run" in report_text
    assert "Run ID" in report_text
    assert "## Retriever Metrics" in report_text
    assert "## Artifacts" in report_text


def test_run_full_evaluation_ignores_non_numeric_ragas_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline_without_init()
    pipeline.rag_pipeline = type(
        "FakeRagPipeline",
        (),
        {
            "run": staticmethod(
                lambda question: {
                    "retrieved_docs": [Document(page_content=f"ctx-{question}", metadata={})],
                    "retrieved_scores": [0.85],
                    "answer": f"ans-{question}",
                }
            )
        },
    )()
    pipeline.load_test_cases = lambda _: [
        QATestCase("q1", "a1", "ctx-q1", "factoid", "therapy", "", [], {})
    ]
    pipeline.evaluate_with_ragas = lambda results: pd.DataFrame(
        [
            {
                "faithfulness": 0.91,
                "question": results[0].question,
                "token_level_debug": [0.1, 0.2, 0.3],
                "question_type": results[0].question_type,
                "section_type": results[0].section_type,
                "test_case_id": results[0].test_case_id,
            }
        ]
    )
    pipeline.calculate_retrieval_metrics = lambda *_args, **_kwargs: {
        "hit_rate": 1.0,
        "mrr": 1.0,
        "avg_score": 0.85,
        "recall_at_k": 1.0,
        "precision_at_k": 1.0,
        "k": 5,
    }
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: path.touch())

    pipeline.run_full_evaluation(
        dataset_file=Path("ignored.jsonl"),
        output_dir=tmp_path,
        sample_size=None,
        k=5,
    )

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    case_rows = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(case_rows) == 1
    assert case_rows[0]["status"] == "ok"
    assert case_rows[0]["ragas_scores"] == {"faithfulness": 0.91}


def test_run_full_evaluation_tracks_rag_and_ragas_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline_without_init()
    calls = {"n": 0}

    def rag_run(question: str) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rag boom")
        return {
            "retrieved_docs": [Document(page_content=f"ctx-{question}", metadata={})],
            "retrieved_scores": [0.8],
            "answer": f"ans-{question}",
        }

    pipeline.rag_pipeline = type("FakeRagPipeline", (), {"run": staticmethod(rag_run)})()
    test_cases = [
        QATestCase("q1", "a1", "ctx-q1", "factoid", "therapy", "", [], {}),
        QATestCase("q2", "a2", "ctx-q2", "procedural", "diagnostics", "", [], {}),
    ]
    pipeline.load_test_cases = lambda _: test_cases

    eval_calls = {"n": 0}

    def evaluate_single(results: list[RAGTestResult]) -> pd.DataFrame:
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            raise RuntimeError("ragas boom")
        return pd.DataFrame(
            [
                {
                    "faithfulness": 0.95,
                    "question_type": results[0].question_type,
                    "section_type": results[0].section_type,
                    "test_case_id": results[0].test_case_id,
                }
            ]
        )

    pipeline.evaluate_with_ragas = evaluate_single
    pipeline.calculate_retrieval_metrics = lambda *_args, **_kwargs: {
        "hit_rate": 1.0,
        "mrr": 1.0,
        "avg_score": 0.8,
        "recall_at_k": 1.0,
        "precision_at_k": 1.0,
        "k": 5,
    }
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: path.touch())

    pipeline.run_full_evaluation(
        dataset_file=Path("ignored.jsonl"),
        output_dir=tmp_path,
        sample_size=None,
        k=5,
    )

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    errors_file = run_dir / "errors.jsonl"
    assert errors_file.exists()
    rows = [json.loads(line) for line in errors_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    stages = {row["stage"] for row in rows}
    assert stages == {"rag", "ragas"}


def test_run_retriever_evaluation_writes_artifacts(tmp_path: Path) -> None:
    pipeline = _pipeline_without_init()
    pipeline.run_retriever_pipeline = lambda *_args, **_kwargs: [
        RAGTestResult(
            question="q",
            ground_truth_answer="a",
            ground_truth_context="ctx",
            retrieved_contexts=[Document(page_content="ctx", metadata={})],
            retrieval_scores=[0.9],
            generated_answer="",
            question_type="factoid",
            section_type="therapy",
            test_case_id=0,
        )
    ]
    pipeline.load_test_cases = lambda _: [
        QATestCase("q", "a", "ctx", "factoid", "therapy", "", [], {})
    ]
    pipeline.calculate_retrieval_metrics = lambda *_args, **_kwargs: {
        "hit_rate": 1.0,
        "mrr": 1.0,
        "avg_score": 0.9,
        "recall_at_k": 1.0,
        "precision_at_k": 0.2,
        "k": 5,
    }

    pipeline.run_retriever_evaluation(
        dataset_file=Path("ignored.jsonl"),
        output_dir=tmp_path,
        sample_size=None,
        k=5,
    )

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "retrieval_cases.jsonl").exists()
    assert (run_dir / "summary_metrics.json").exists()
    report_file = run_dir / "report.md"
    assert report_file.exists()
    report_text = report_file.read_text(encoding="utf-8")
    assert "## Run" in report_text
    assert "## Retriever Metrics" in report_text
    assert "## Artifacts" in report_text
