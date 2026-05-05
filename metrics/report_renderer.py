from typing import Any

import pandas as pd


def render_validation_report(
    *,
    manifest: dict[str, Any],
    summary_metrics: dict[str, Any],
    ragas_df: pd.DataFrame | None,
    mode: str,
    missed_retrieval_case_ids: list[int],
) -> str:
    """
    Render a human-readable markdown report for validation artifacts

    Args:
        manifest: Manifest for the run
        summary_metrics: Summary metrics for the run
        ragas_df: RAGAS dataframe for the run
        mode: Mode of the run
        missed_retrieval_case_ids: List of case IDs that were missed in retrieval
    Returns:
        String containing the rendered report
    """
    lines: list[str] = ["# RAG Validation Report\n\n"]
    lines.extend(_render_run_section(manifest=manifest, mode=mode))
    lines.extend(_render_rag_config_section(manifest=manifest, mode=mode))
    lines.extend(_render_execution_summary_section(summary_metrics=summary_metrics))
    lines.extend(_render_retriever_metrics_section(summary_metrics=summary_metrics))
    lines.extend(_render_text_match_retriever_metrics_section(summary_metrics=summary_metrics))
    if mode == "full":
        lines.extend(_render_ragas_metrics_section(summary_metrics=summary_metrics))
        lines.extend(_render_breakdown_sections(ragas_df=ragas_df))
    lines.extend(_render_problem_cases_section(missed_retrieval_case_ids=missed_retrieval_case_ids))
    lines.extend(_render_artifacts_section(mode=mode))
    return "".join(lines)


def _render_run_section(*, manifest: dict[str, Any], mode: str) -> list[str]:
    """
    Render the run section of the report

    Args:
        manifest: Manifest for the run
        mode: Mode of the run
    Returns:
        List of strings containing the rendered run section
    """
    return [
        "## Run\n",
        f"- Run ID: `{manifest['run_id']}`\n",
        f"- Mode: `{mode}`\n",
        f"- Timestamp (UTC): `{manifest['timestamp_utc']}`\n",
        f"- Dataset: `{manifest['paths']['dataset_file']}`\n",
        f"- Sample size: `{manifest['run_parameters']['sample_size']}`\n",
        f"- K: `{manifest['run_parameters']['k']}`\n",
    ]


def _render_rag_config_section(*, manifest: dict[str, Any], mode: str) -> list[str]:
    """
    Render the RAG configuration section of the report

    Args:
        manifest: Manifest for the run
    Returns:
        List of strings containing the rendered RAG configuration section
    """
    rag_cfg = manifest["rag_config"]
    retrieval_cfg = rag_cfg["retrieval"]
    lines = [
        "## RAG Configuration\n",
        f"- Embedding: `{rag_cfg['embedding_model']}`\n",
        f"- Reranker: `{rag_cfg['reranker_model']}`\n",
        f"- Chunking: size={rag_cfg['chunk_size']}, overlap={rag_cfg['chunk_overlap']}\n",
        "- Retrieval: "
        f"mode={retrieval_cfg['search_mode']}, fusion={retrieval_cfg['fusion_method']}, "
        f"sparse_top_k={retrieval_cfg['sparse_top_k']}, "
        f"dense_top_k={retrieval_cfg['dense_top_k']}, "
        f"hybrid_top_k={retrieval_cfg['hybrid_top_k']}\n",
        f"- Embedding query instruction: `{rag_cfg['embedding_query_instruction']}`\n",
        f"- Reranker query instruction: `{rag_cfg['reranker_query_instruction']}`\n",
        f"- Qdrant collection: `{rag_cfg['qdrant_collection']}`\n\n",
    ]
    if mode == "full":
        lines.insert(1, f"- LLM: `{rag_cfg['llm_model']}`\n")
    return lines


def _render_execution_summary_section(*, summary_metrics: dict[str, Any]) -> list[str]:
    """
    Render the execution summary section of the report

    Args:
        summary_metrics: Summary metrics for the run
    Returns:
        List of strings containing the rendered execution summary section
    """
    return [
        "## Execution Summary\n",
        f"- Total loaded cases: {summary_metrics['total_loaded_cases']}\n",
        f"- Evaluated cases: {summary_metrics['evaluated_cases']}\n",
        f"- RAG success cases: {summary_metrics['rag_success_cases']}\n",
        f"- RAG errors: {summary_metrics['rag_errors']}\n",
        f"- RAGAS errors: {summary_metrics['ragas_errors']}\n\n",
    ]


def _render_retriever_metrics_section(*, summary_metrics: dict[str, Any]) -> list[str]:
    """
    Render the retriever metrics section of the report

    Args:
        summary_metrics: Summary metrics for the run
    Returns:
        List of strings containing the rendered retriever metrics section
    """
    retrieval_metrics = summary_metrics["retrieval_metrics"]
    rk = retrieval_metrics.get("k")
    return [
        "## Retriever Metrics\n",
        f"- Hit Rate (top-{rk}): {retrieval_metrics['hit_rate']:.3f}\n",
        f"- MRR: {retrieval_metrics['mrr']:.3f}\n",
        f"- Recall@{rk}: {retrieval_metrics['recall_at_k']:.3f}\n",
        f"- Precision@{rk}: {retrieval_metrics['precision_at_k']:.3f}\n",
        f"- Average score top-1: {retrieval_metrics['avg_score']:.3f}\n\n",
    ]


def _render_text_match_retriever_metrics_section(*, summary_metrics: dict[str, Any]) -> list[str]:
    """
    Render the text match retriever metrics section of the report

    Args:
        summary_metrics: Summary metrics for the run
    Returns:
        List of strings containing the rendered text match retriever metrics section
    """
    retrieval_metrics = summary_metrics.get("text_match_retrieval_metrics")
    if not retrieval_metrics:
        return []

    rk = retrieval_metrics.get("text_match_k")
    return [
        "## Text Matcher Retriever Metrics\n",
        f"- Hit Rate (top-{rk}): {retrieval_metrics['text_match_hit_rate']:.3f}\n",
        f"- MRR: {retrieval_metrics['text_match_mrr']:.3f}\n",
        f"- Recall@{rk}: {retrieval_metrics['text_match_recall_at_k']:.3f}\n",
        f"- Precision@{rk}: {retrieval_metrics['text_match_precision_at_k']:.3f}\n",
        f"- Average score top-1: {retrieval_metrics['text_match_avg_score']:.3f}\n\n",
    ]


def _render_ragas_metrics_section(*, summary_metrics: dict[str, Any]) -> list[str]:
    """
    Render the RAGAS metrics section of the report

    Args:
        summary_metrics: Summary metrics for the run
    Returns:
        List of strings containing the rendered RAGAS metrics section
    """
    lines: list[str] = ["## RAGAS Metrics\n"]
    ragas_summary = summary_metrics.get("ragas_metrics", {})
    if not ragas_summary:
        return [*lines, "- No successful RAGAS metric rows.\n\n"]

    for metric_name, metric_data in ragas_summary.items():
        lines.extend(
            [
                f"### {metric_name}\n",
                f"- Count: {metric_data['count']}\n",
                f"- Mean: {metric_data['mean']:.3f}\n",
                f"- Median: {metric_data['median']:.3f}\n",
                f"- Std: {metric_data['std']:.3f}\n",
                f"- Min: {metric_data['min']:.3f}\n",
                f"- Max: {metric_data['max']:.3f}\n\n",
            ]
        )
    return lines


def _render_breakdown_sections(*, ragas_df: pd.DataFrame | None) -> list[str]:
    """
    Render the breakdown sections of the report

    Args:
        ragas_df: RAGAS dataframe for the run
    Returns:
        List of strings containing the rendered breakdown sections
    """
    if ragas_df is None or ragas_df.empty:
        return []

    lines: list[str] = ["## Breakdown by Question Type\n"]
    for qtype in ragas_df["question_type"].dropna().unique():
        subset = ragas_df[ragas_df["question_type"] == qtype]
        lines.append(f"- `{qtype}` (n={len(subset)})\n")
    lines.append("\n")

    lines.append("## Breakdown by Section Type\n")
    for stype in ragas_df["section_type"].dropna().unique():
        subset = ragas_df[ragas_df["section_type"] == stype]
        lines.append(f"- `{stype}` (n={len(subset)})\n")
    lines.append("\n")
    return lines


def _render_problem_cases_section(*, missed_retrieval_case_ids: list[int]) -> list[str]:
    """
    Render the problem cases section of the report

    Args:
        missed_retrieval_case_ids: List of case IDs that were missed in retrieval
    Returns:
        List of strings containing the rendered problem cases section
    """
    lines = [
        "## Problem Cases\n",
        f"- Missed retrieval cases: {len(missed_retrieval_case_ids)}\n",
    ]
    if missed_retrieval_case_ids:
        ids_preview = ", ".join(str(case_id) for case_id in missed_retrieval_case_ids[:20])
        lines.append(f"- Missed test_case_id preview: {ids_preview}\n")
    lines.append("\n")
    return lines


def _render_artifacts_section(*, mode: str) -> list[str]:
    """
    Render the artifacts section of the report

    Args:
        mode: Mode of the run
    Returns:
        List of strings containing the rendered artifacts section
    """
    lines = [
        "## Artifacts\n",
        "- `run_manifest.json`\n",
    ]
    if mode == "full":
        lines.extend(["- `cases.jsonl`\n", "- `ragas_scores.parquet`\n"])
    else:
        lines.append("- `retrieval_cases.jsonl`\n")
    lines.extend(["- `errors.jsonl`\n", "- `summary_metrics.json`\n", "- `report.md`\n"])
    return lines
