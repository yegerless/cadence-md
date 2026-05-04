import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document

from commands import build_project_cli_parser
from metrics.evaluation_pipeline import RAGEvaluationPipeline
from metrics.schemas import QATestCase, RAGTestResult


def _pipeline_without_init() -> RAGEvaluationPipeline:
    pipeline = object.__new__(RAGEvaluationPipeline)
    pipeline.ragas_metrics = []
    return pipeline


def _ranked(
    doc: Document,
    *,
    rank: int = 1,
    retrieval_score: float | None = None,
    rerank_score: float | None = None,
    final_score: float | None = None,
) -> dict:
    """Build a ``RankedDocument`` dict for tests."""
    return {
        "rank": rank,
        "doc": doc,
        "retrieval_score": retrieval_score,
        "rerank_score": rerank_score,
        "final_score": final_score,
        "chunk_id": None,
        "section_id": None,
    }


def test_load_test_cases_skips_malformed_rows(tmp_path: Path) -> None:
    dataset_file = tmp_path / "qa_dataset.jsonl"
    valid_row = {
        "question": "Q1",
        "answer": "A1",
        "context": "C1",
        "question_type": "factoid",
        "section_type": "diagnostics",
        "section_id": "section-1",
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
    assert cases[0].section_id == "section-1"


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


def test_calculate_retrieval_metrics_uses_section_id_not_text_fallback() -> None:
    pipeline = _pipeline_without_init()
    result = RAGTestResult(
        question="Q",
        ground_truth_answer="A",
        ground_truth_context="эталонный контекст полностью совпадает с первым документом",
        retrieved_contexts=[
            Document(
                page_content="эталонный контекст полностью совпадает с первым документом",
                metadata={"section_id": "wrong-section"},
            ),
            Document(
                page_content="совершенно другой текст из правильной секции",
                metadata={"section_id": "expected-section"},
            ),
        ],
        retrieval_scores=[0.9, 0.7],
        generated_answer="",
        question_type="factoid",
        section_type="therapy",
        test_case_id=0,
        ground_truth_section_id="expected-section",
    )

    metrics = pipeline.calculate_retrieval_metrics([result], k=2)

    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["recall_at_k"] == 1.0
    assert metrics["precision_at_k"] == 0.5


def test_calculate_retrieval_metrics_without_section_id_counts_miss() -> None:
    pipeline = _pipeline_without_init()
    result = RAGTestResult(
        question="Q",
        ground_truth_answer="A",
        ground_truth_context="эталонный контекст полностью совпадает",
        retrieved_contexts=[
            Document(page_content="эталонный контекст полностью совпадает", metadata={}),
        ],
        retrieval_scores=[0.9],
        generated_answer="",
        question_type="factoid",
        section_type="therapy",
        test_case_id=0,
    )

    metrics = pipeline.calculate_retrieval_metrics([result], k=1)

    assert metrics["hit_rate"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["recall_at_k"] == 0.0


def test_calculate_text_match_metrics_uses_prefixed_keys() -> None:
    pipeline = _pipeline_without_init()
    result = RAGTestResult(
        question="Q",
        ground_truth_answer="A",
        ground_truth_context=(
            "пациенту рекомендуется ацетилсалициловая кислота в дозе 75 мг ежедневно"
        ),
        retrieved_contexts=[
            Document(
                page_content=(
                    "ацетилсалициловая кислота в дозе 75 мг ежедневно назначается пациенту"
                ),
                metadata={},
            ),
        ],
        retrieval_scores=[0.9],
        generated_answer="",
        question_type="factoid",
        section_type="therapy",
        test_case_id=0,
    )

    metrics = pipeline.calculate_retrieval_metrics(
        [result],
        k=1,
        matcher=pipeline.text_matcher_matches,
        metric_prefix="text_match_",
    )

    assert metrics["text_match_k"] == 1
    assert metrics["text_match_hit_rate"] == 1.0
    assert "hit_rate" not in metrics


def test_metrics_parser_rejects_non_positive_k(tmp_path: Path) -> None:
    parser = build_project_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "metrics-eval-retriever",
                "--dataset-file",
                "data/metrics_evaluation_datasets/qa_dataset.jsonl",
                "--output-dir",
                str(tmp_path),
                "--k",
                "0",
            ]
        )


def test_metrics_parser_accepts_valid_full_args(tmp_path: Path) -> None:
    parser = build_project_cli_parser()
    args = parser.parse_args(
        [
            "metrics-eval-full",
            "--dataset-file",
            "data/metrics_evaluation_datasets/qa_dataset.jsonl",
            "--output-dir",
            str(tmp_path),
            "--sample-size",
            "10",
            "--k",
            "7",
            "--enable-text-matcher-metrics",
        ]
    )
    assert args.command == "metrics-eval-full"
    assert args.sample_size == 10
    assert args.k == 7
    assert args.enable_text_matcher_metrics is True


def test_metrics_parser_full_uses_default_k(tmp_path: Path) -> None:
    parser = build_project_cli_parser()
    args = parser.parse_args(
        [
            "metrics-eval-full",
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
                    "ranked_docs": [
                        _ranked(
                            Document(page_content=f"ctx-{question}", metadata={}),
                            retrieval_score=0.9,
                            rerank_score=0.9,
                            final_score=0.9,
                        )
                    ],
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
                    "ranked_docs": [
                        _ranked(
                            Document(page_content=f"ctx-{question}", metadata={}),
                            retrieval_score=0.85,
                            rerank_score=0.85,
                            final_score=0.85,
                        )
                    ],
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


def test_run_full_evaluation_writes_optional_text_match_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline_without_init()
    pipeline.rag_pipeline = type(
        "FakeRagPipeline",
        (),
        {
            "run": staticmethod(
                lambda _question: {
                    "ranked_docs": [
                        _ranked(
                            Document(
                                page_content="ацетилсалициловая кислота в дозе 75 мг ежедневно",
                                metadata={"section_id": "wrong"},
                            ),
                            retrieval_score=0.85,
                            rerank_score=0.85,
                            final_score=0.85,
                        )
                    ],
                    "answer": "answer",
                }
            )
        },
    )()
    pipeline.load_test_cases = lambda _: [
        QATestCase(
            "q1",
            "a1",
            "ацетилсалициловая кислота в дозе 75 мг ежедневно",
            "factoid",
            "therapy",
            "",
            [],
            {},
            "expected",
        )
    ]
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
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: path.touch())

    pipeline.run_full_evaluation(
        dataset_file=Path("ignored.jsonl"),
        output_dir=tmp_path,
        sample_size=None,
        k=1,
        enable_text_matcher_metrics=True,
    )

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    summary = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    assert summary["retrieval_metrics"]["hit_rate"] == 0.0
    assert summary["text_match_retrieval_metrics"]["text_match_hit_rate"] == 1.0

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_parameters"]["enable_text_matcher_metrics"] is True


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
            "ranked_docs": [
                _ranked(
                    Document(page_content=f"ctx-{question}", metadata={}),
                    retrieval_score=0.8,
                    rerank_score=0.8,
                    final_score=0.8,
                )
            ],
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


def test_run_retriever_evaluation_writes_optional_text_match_metrics(tmp_path: Path) -> None:
    pipeline = _pipeline_without_init()
    pipeline.run_retriever_pipeline = lambda *_args, **_kwargs: [
        RAGTestResult(
            question="q",
            ground_truth_answer="a",
            ground_truth_context="ацетилсалициловая кислота в дозе 75 мг ежедневно",
            retrieved_contexts=[
                Document(
                    page_content="ацетилсалициловая кислота в дозе 75 мг ежедневно",
                    metadata={"section_id": "wrong"},
                )
            ],
            retrieval_scores=[0.9],
            generated_answer="",
            question_type="factoid",
            section_type="therapy",
            test_case_id=0,
            ground_truth_section_id="expected",
        )
    ]
    pipeline.load_test_cases = lambda _: [
        QATestCase(
            "q",
            "a",
            "ацетилсалициловая кислота в дозе 75 мг ежедневно",
            "factoid",
            "therapy",
            "",
            [],
            {},
            "expected",
        )
    ]

    pipeline.run_retriever_evaluation(
        dataset_file=Path("ignored.jsonl"),
        output_dir=tmp_path,
        sample_size=None,
        k=1,
        enable_text_matcher_metrics=True,
    )

    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    summary = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    assert summary["retrieval_metrics"]["hit_rate"] == 0.0
    assert summary["text_match_retrieval_metrics"]["text_match_hit_rate"] == 1.0

    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Text Matcher Retriever Metrics" in report_text

    retrieval_case = json.loads(
        (run_dir / "retrieval_cases.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert retrieval_case["matched"] is False
    assert retrieval_case["text_match_matched"] is True


def test_normalize_match_text_yo_and_punctuation() -> None:
    norm = RAGEvaluationPipeline._normalize_match_text("  Ёжик,  тест!!  ")
    assert "ежик" in norm
    assert "тест" in norm
    assert norm == norm.strip()


def test_meaningful_tokens_filters_stopwords_and_short() -> None:
    text = "для ab я x очень длинное слово здесь"
    tokens = RAGEvaluationPipeline._meaningful_tokens(
        RAGEvaluationPipeline._normalize_match_text(text)
    )
    assert "для" not in tokens
    assert "ab" not in tokens
    assert "длинное" in tokens


def test_char_ngrams_short_string_returns_empty() -> None:
    assert RAGEvaluationPipeline._char_ngrams("abcd", n=5) == set()


def test_char_ngrams_five_window() -> None:
    ngrams = RAGEvaluationPipeline._char_ngrams("abcdefghij", n=5)
    assert len(ngrams) == 6
    assert "abcde" in ngrams
    assert "fghij" in ngrams


def test_char_ngram_match_identical_long_text() -> None:
    text = "абвгдежзийклмнопрстуфхцчшщъыьэюя" * 3
    assert RAGEvaluationPipeline._char_ngram_match(
        gt_text=text,
        retrieved_text=text,
        gt_coverage=0.9,
    )


def test_contexts_match_substring_when_long_enough() -> None:
    gt = "x" * 40
    retrieved = gt + " хвост дополнительный текст"
    assert RAGEvaluationPipeline.contexts_match(gt, retrieved)


def test_contexts_match_true_via_token_coverage_threshold() -> None:
    shared = " ".join(f"wtoken{i}" for i in range(5))
    gt = shared + " " + " ".join(f"gt{i}" for i in range(3))
    retrieved = shared + " " + " ".join(f"rt{i}" for i in range(3))
    assert RAGEvaluationPipeline.contexts_match(gt, retrieved)


def test_contexts_match_false_when_token_overlap_below_five() -> None:
    gt = "alphaone bravotwo charlie three delta four"
    retrieved = "zzzzzz zzzzzz zzzzzz zzzzzz zzzzzz"
    assert not RAGEvaluationPipeline.contexts_match(gt, retrieved)


def test_contexts_match_false_on_whitespace_only() -> None:
    assert not RAGEvaluationPipeline.contexts_match("   \n\t", "  ")


def test_contexts_match_ngram_fallback_when_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = " ".join(f"xtoken{i}" for i in range(5))
    gt = shared + " " + " ".join(f"ga{i}" for i in range(6))
    retrieved = shared + " " + " ".join(f"rb{i}" for i in range(6))
    monkeypatch.setattr(RAGEvaluationPipeline, "_char_ngram_match", lambda **_: True)
    assert RAGEvaluationPipeline.contexts_match(gt, retrieved)


def test_section_ids_match_requires_non_empty_expected() -> None:
    result = RAGTestResult(
        question="Q",
        ground_truth_answer="A",
        ground_truth_context="c",
        retrieved_contexts=[],
        retrieval_scores=[],
        generated_answer="",
        question_type="f",
        section_type="s",
        test_case_id=0,
        ground_truth_section_id="",
    )
    doc = Document(page_content="x", metadata={"section_id": ""})
    assert not RAGEvaluationPipeline.section_ids_match(result, doc)


def test_section_ids_match_strips_whitespace() -> None:
    result = RAGTestResult(
        question="Q",
        ground_truth_answer="A",
        ground_truth_context="c",
        retrieved_contexts=[],
        retrieval_scores=[],
        generated_answer="",
        question_type="f",
        section_type="s",
        test_case_id=0,
        ground_truth_section_id="  sec-a  ",
    )
    doc = Document(page_content="x", metadata={"section_id": "sec-a"})
    assert RAGEvaluationPipeline.section_ids_match(result, doc)


def test_matched_rank_one_based_and_respects_k() -> None:
    docs = [
        Document("a", metadata={"i": 0}),
        Document("b", metadata={"i": 1}),
        Document("c", metadata={"i": 2}),
    ]
    result = RAGTestResult(
        question="Q",
        ground_truth_answer="A",
        ground_truth_context="c",
        retrieved_contexts=docs,
        retrieval_scores=[0.1, 0.2, 0.3],
        generated_answer="",
        question_type="f",
        section_type="s",
        test_case_id=0,
    )

    def matcher(_r: RAGTestResult, d: Document) -> bool:
        return d.metadata.get("i") == 1

    assert RAGEvaluationPipeline._matched_rank(result, k=3, matcher=matcher) == 2
    assert RAGEvaluationPipeline._matched_rank(result, k=1, matcher=matcher) is None

    def matcher_none(_r: RAGTestResult, _d: Document) -> bool:
        return False

    assert RAGEvaluationPipeline._matched_rank(result, k=3, matcher=matcher_none) is None


def test_serialize_retrieved_docs_scores_optional() -> None:
    result = RAGTestResult(
        question="Q",
        ground_truth_answer="A",
        ground_truth_context="c",
        retrieved_contexts=[
            Document("a", metadata={"k": 1}),
            Document("b", metadata={"k": 2}),
        ],
        retrieval_scores=[0.5],
        generated_answer="",
        question_type="f",
        section_type="s",
        test_case_id=0,
    )
    rows = RAGEvaluationPipeline._serialize_retrieved_docs(result)
    assert rows[0]["score"] == 0.5
    assert "score" not in rows[1]
    assert rows[1]["metadata"]["k"] == 2


def test_convert_to_ragas_format_maps_columns() -> None:
    pipeline = _pipeline_without_init()
    results = [
        RAGTestResult(
            question="q1",
            ground_truth_answer="gt",
            ground_truth_context="ctx",
            retrieved_contexts=[
                Document("r1", metadata={}),
                Document("r2", metadata={}),
            ],
            retrieval_scores=[0.1, 0.2],
            generated_answer="gen",
            question_type="factoid",
            section_type="therapy",
            test_case_id=0,
        )
    ]
    ds = pipeline.convert_to_ragas_format(results)
    assert ds["question"] == ["q1"]
    assert ds["answer"] == ["gen"]
    assert ds["ground_truth"] == ["gt"]
    assert ds["contexts"] == [["r1", "r2"]]


def test_evaluate_with_ragas_empty_results_dataframe() -> None:
    pipeline = _pipeline_without_init()
    df = pipeline.evaluate_with_ragas([])
    assert list(df.columns) == ["question_type", "section_type", "test_case_id"]
    assert len(df) == 0


def test_run_rag_pipeline_sample_and_error_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline_without_init()

    def rag_run(question: str) -> dict:
        if question == "q1":
            raise RuntimeError("fail case")
        return {
            "ranked_docs": [
                _ranked(
                    Document(page_content=question, metadata={}),
                    retrieval_score=0.5,
                    rerank_score=0.5,
                    final_score=0.5,
                )
            ],
            "answer": f"a-{question}",
        }

    pipeline.rag_pipeline = type("R", (), {"run": staticmethod(rag_run)})()
    cases = [QATestCase(f"q{i}", f"a{i}", "c", "f", "s", "", [], {}) for i in range(10)]
    monkeypatch.setattr(
        "metrics.evaluation_pipeline.random.sample",
        lambda population, k: list(population)[:k],
    )
    results = pipeline.run_rag_pipeline(cases, sample_size=3)
    assert len(results) == 2
    assert {r.test_case_id for r in results} == {0, 2}
    assert results[0].question == "q0"
    assert results[1].question == "q2"


def test_run_retriever_pipeline_scores_and_errors() -> None:
    pipeline = _pipeline_without_init()
    calls = {"n": 0}

    def retrieve_node(state):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("retrieve fail")
        return {
            **state,
            "ranked_docs": [
                _ranked(
                    Document("d", metadata={}),
                    retrieval_score=0.1,
                    final_score=0.1,
                )
            ],
        }

    def reranker_node(state):
        ranked = state["ranked_docs"]
        return {
            **state,
            "ranked_docs": [
                _ranked(
                    rd["doc"],
                    rank=rd["rank"],
                    retrieval_score=rd["retrieval_score"],
                    rerank_score=0.99,
                    final_score=0.99,
                )
                for rd in ranked
            ],
        }

    pipeline.rag_pipeline = type(
        "R",
        (),
        {
            "retrieve_node": staticmethod(retrieve_node),
            "reranker_node": staticmethod(reranker_node),
        },
    )()
    cases = [
        QATestCase("q0", "a0", "c0", "f", "s", "", [], {}),
        QATestCase("q1", "a1", "c1", "f", "s", "", [], {}),
        QATestCase("q2", "a2", "c2", "f", "s", "", [], {}),
    ]
    results = pipeline.run_retriever_pipeline(cases, sample_size=None)
    assert len(results) == 2
    assert results[0].retrieval_scores == [0.99]
    assert results[0].test_case_id == 0
    assert results[1].question == "q2"


def test_run_retriever_pipeline_uses_ranked_docs_final_score() -> None:
    """``run_retriever_pipeline`` must read ``final_score`` from ``ranked_docs`` after rerank."""
    pipeline = _pipeline_without_init()

    def retrieve_node(state):
        return {
            **state,
            "ranked_docs": [
                _ranked(
                    Document("d", metadata={}),
                    retrieval_score=0.42,
                    final_score=0.42,
                )
            ],
        }

    def reranker_node(state):
        # Rerank fallback path: keep ranked_docs untouched (final_score == retrieval_score).
        return state

    pipeline.rag_pipeline = type(
        "R",
        (),
        {
            "retrieve_node": staticmethod(retrieve_node),
            "reranker_node": staticmethod(reranker_node),
        },
    )()
    cases = [QATestCase("q", "a", "c", "f", "s", "", [], {})]
    results = pipeline.run_retriever_pipeline(cases)
    assert results[0].retrieval_scores == [0.42]


def test_run_full_evaluation_uses_ranked_docs_aligned_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full RAG path must record final scores aligned with the post-rerank doc order."""
    pipeline = _pipeline_without_init()
    captured: dict[str, list[float]] = {}

    def rag_run(question: str) -> dict:
        d1 = Document(page_content=f"ctx1-{question}", metadata={})
        d2 = Document(page_content=f"ctx2-{question}", metadata={})
        return {
            "ranked_docs": [
                _ranked(d1, rank=1, retrieval_score=0.2, rerank_score=0.9, final_score=0.9),
                _ranked(d2, rank=2, retrieval_score=0.1, rerank_score=0.5, final_score=0.5),
            ],
            "answer": f"ans-{question}",
        }

    pipeline.rag_pipeline = type("R", (), {"run": staticmethod(rag_run)})()
    pipeline.load_test_cases = lambda _: [
        QATestCase("q1", "a1", "ctx1-q1", "factoid", "therapy", "", [], {})
    ]

    def fake_evaluate(results: list[RAGTestResult]) -> pd.DataFrame:
        captured["scores"] = list(results[0].retrieval_scores)
        return pd.DataFrame(
            [
                {
                    "faithfulness": 1.0,
                    "question_type": results[0].question_type,
                    "section_type": results[0].section_type,
                    "test_case_id": results[0].test_case_id,
                }
            ]
        )

    pipeline.evaluate_with_ragas = fake_evaluate
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

    assert captured["scores"] == [0.9, 0.5]
