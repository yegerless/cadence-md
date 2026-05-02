from typing import Any

import pandas as pd


def build_summary_metrics(
    *,
    mode: str,
    total_loaded_cases: int,
    evaluated_cases: int,
    rag_success_cases: int,
    retrieval_metrics: dict[str, float | int],
    ragas_df: pd.DataFrame | None,
    ragas_metric_names: list[str],
    rag_errors: int,
    ragas_errors: int,
) -> dict[str, Any]:
    """Build summary payload for summary_metrics.json artifact."""
    summary: dict[str, Any] = {
        "mode": mode,
        "total_loaded_cases": total_loaded_cases,
        "evaluated_cases": evaluated_cases,
        "rag_success_cases": rag_success_cases,
        "rag_errors": rag_errors,
        "ragas_errors": ragas_errors,
        "retrieval_metrics": retrieval_metrics,
    }
    ragas_summary = _build_ragas_summary(ragas_df=ragas_df, ragas_metric_names=ragas_metric_names)
    if ragas_summary:
        summary["ragas_metrics"] = ragas_summary
    return summary


def _build_ragas_summary(
    *, ragas_df: pd.DataFrame | None, ragas_metric_names: list[str]
) -> dict[str, dict[str, float | int]]:
    if ragas_df is None or ragas_df.empty:
        return {}

    ragas_summary: dict[str, dict[str, float | int]] = {}
    for metric_name in ragas_metric_names:
        if metric_name not in ragas_df.columns:
            continue
        values = ragas_df[metric_name].dropna()
        if values.empty:
            continue
        ragas_summary[metric_name] = _metric_stats(values)
    return ragas_summary


def _metric_stats(values: pd.Series) -> dict[str, float | int]:
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
