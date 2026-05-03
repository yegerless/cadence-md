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
    text_match_retrieval_metrics: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """
    Build summary payload for summary_metrics.json artifact

    Args:
        mode: Mode of the run (retriever or full)
        total_loaded_cases: Total number of loaded test cases
        evaluated_cases: Number of evaluated test cases
        rag_success_cases: Number of successful RAG cases
        retrieval_metrics: Retrieval metrics
        ragas_df: RAGAS dataframe
        ragas_metric_names: List of RAGAS metric names
        rag_errors: Number of RAG errors
        ragas_errors: Number of RAGAS errors
        text_match_retrieval_metrics: Text match retrieval metrics
    Returns:
        Dictionary containing the summary payload
    """
    # Build the summary payload
    summary: dict[str, Any] = {
        "mode": mode,
        "total_loaded_cases": total_loaded_cases,
        "evaluated_cases": evaluated_cases,
        "rag_success_cases": rag_success_cases,
        "rag_errors": rag_errors,
        "ragas_errors": ragas_errors,
        "retrieval_metrics": retrieval_metrics,
    }
    # Add text match retrieval metrics if they are provided
    if text_match_retrieval_metrics is not None:
        summary["text_match_retrieval_metrics"] = text_match_retrieval_metrics
    ragas_summary = _build_ragas_summary(ragas_df=ragas_df, ragas_metric_names=ragas_metric_names)
    # Add RAGAS metrics if they are provided
    if ragas_summary:
        summary["ragas_metrics"] = ragas_summary
    return summary


def _build_ragas_summary(
    *, ragas_df: pd.DataFrame | None, ragas_metric_names: list[str]
) -> dict[str, dict[str, float | int]]:
    """
    Build RAGAS summary

    Args:
        ragas_df: RAGAS dataframe
        ragas_metric_names: List of RAGAS metric names
    Returns:
        Dictionary containing the RAGAS summary
    """
    # Check if the RAGAS dataframe is provided and is not empty
    if ragas_df is None or ragas_df.empty:
        return {}
    # Build the RAGAS summary
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
    """
    Calculate metric statistics

    Args:
        values: Series of values
    Returns:
        Dictionary containing the metric statistics
    """
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
