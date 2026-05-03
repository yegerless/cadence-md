import pandas as pd

from metrics.summary_builder import build_summary_metrics


def test_build_summary_metrics_with_ragas_scores() -> None:
    ragas_df = pd.DataFrame(
        [
            {"faithfulness": 0.8, "answer_correctness": 0.7},
            {"faithfulness": 0.6, "answer_correctness": 0.9},
        ]
    )
    summary = build_summary_metrics(
        mode="full",
        total_loaded_cases=10,
        evaluated_cases=8,
        rag_success_cases=8,
        retrieval_metrics={
            "k": 5,
            "hit_rate": 0.75,
            "mrr": 0.64,
            "recall_at_k": 0.8,
            "precision_at_k": 0.2,
            "avg_score": 0.91,
        },
        ragas_df=ragas_df,
        ragas_metric_names=["faithfulness", "answer_correctness", "missing_metric"],
        rag_errors=1,
        ragas_errors=1,
    )

    assert summary["mode"] == "full"
    assert summary["retrieval_metrics"]["k"] == 5
    assert "ragas_metrics" in summary
    assert summary["ragas_metrics"]["faithfulness"]["count"] == 2
    assert summary["ragas_metrics"]["faithfulness"]["mean"] == 0.7
    assert summary["ragas_metrics"]["answer_correctness"]["max"] == 0.9
    assert "missing_metric" not in summary["ragas_metrics"]


def test_build_summary_metrics_without_ragas_scores() -> None:
    summary = build_summary_metrics(
        mode="retriever",
        total_loaded_cases=12,
        evaluated_cases=12,
        rag_success_cases=12,
        retrieval_metrics={
            "k": 7,
            "hit_rate": 0.9,
            "mrr": 0.83,
            "recall_at_k": 0.9,
            "precision_at_k": 0.1,
            "avg_score": 0.95,
        },
        ragas_df=None,
        ragas_metric_names=["faithfulness"],
        rag_errors=0,
        ragas_errors=0,
    )

    assert summary["mode"] == "retriever"
    assert summary["total_loaded_cases"] == 12
    assert summary["retrieval_metrics"]["k"] == 7
    assert "ragas_metrics" not in summary


def test_build_summary_metrics_includes_optional_text_match_metrics() -> None:
    summary = build_summary_metrics(
        mode="retriever",
        total_loaded_cases=1,
        evaluated_cases=1,
        rag_success_cases=1,
        retrieval_metrics={
            "k": 5,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "avg_score": 0.5,
        },
        text_match_retrieval_metrics={
            "text_match_k": 5,
            "text_match_hit_rate": 1.0,
            "text_match_mrr": 1.0,
            "text_match_recall_at_k": 1.0,
            "text_match_precision_at_k": 0.2,
            "text_match_avg_score": 0.5,
        },
        ragas_df=None,
        ragas_metric_names=[],
        rag_errors=0,
        ragas_errors=0,
    )

    assert summary["text_match_retrieval_metrics"]["text_match_hit_rate"] == 1.0
