import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from cadence_md.app.settings import settings
from metrics.config import metrics_settings

RAG_GRAPH_PROFILE_FIELDS = (
    "enable_input_guardrails",
    "enable_query_rewriter",
    "enable_context_relevance_grader",
    "enable_answer_formatter",
    "enable_output_guardrails",
    "enable_query_clarification",
    "max_query_rewrite_iterations",
)


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format"""
    return datetime.now(UTC).isoformat()


def build_run_id(mode: str, timestamp_iso: str) -> str:
    """
    Build deterministic run id from mode and timestamp

    Args:
        mode: Mode of the run (retriever or full)
        timestamp_iso: Timestamp in ISO 8601 format
    Returns:
        String containing the run id
    """
    ts = timestamp_iso.replace("-", "").replace(":", "").split(".")[0].replace("+0000", "Z")
    return f"{mode}_{ts.replace('+00', 'Z')}"


def build_run_directory(output_dir: Path, mode: str) -> tuple[Path, str, str]:
    """
    Create and return run directory, run id, and timestamp

    Args:
        output_dir: Path to the output directory (usuallymetrics/results/)
        mode: Mode of the run (retriever or full)
    Returns:
        Tuple containing the run directory, run id, and timestamp
    """
    timestamp_iso = now_utc_iso()
    run_id = build_run_id(mode=mode, timestamp_iso=timestamp_iso)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id, timestamp_iso


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """
    Append a single JSON object to JSONL file

    Args:
        path: Path to the JSONL file
        row: JSON object to append
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_normalize_json_value(row), ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """
    Write JSON payload using UTF-8 and pretty formatting

    Args:
        path: Path to the JSON file
        payload: JSON object to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_normalize_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def document_to_record(document: Document, score: float | None = None) -> dict[str, Any]:
    """
    Serialize LangChain document for artifacts

    Args:
        document: LangChain document
        score: Score of the document
    Returns:
        Dictionary containing the document and score
    """
    record: dict[str, Any] = {
        "page_content": document.page_content,
        "metadata": document.metadata,
    }
    if score is not None:
        record["score"] = score
    return record


def create_run_manifest(
    *,
    mode: str,
    run_id: str,
    timestamp_iso: str,
    output_dir: Path,
    run_dir: Path,
    dataset_file: Path,
    sample_size: int | None,
    k: int | None,
    ragas_metric_names: list[str],
    sample_seed: int | None = None,
    enable_text_matcher_metrics: bool = False,
    workers: int | None = None,
    rag_optional_nodes_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build manifest payload for a validation run

    Args:
        mode: Mode of the run (retriever or full)
        run_id: Run id
        timestamp_iso: Timestamp in ISO 8601 format
        output_dir: Path to the output directory
        run_dir: Path to the run directory
        dataset_file: Path to the dataset file
        sample_size: Number of test cases to evaluate
        k: Number of retrieved documents to evaluate
        ragas_metric_names: List of RAGAS metric names to evaluate
        sample_seed: Random seed used for sampling, when provided
        enable_text_matcher_metrics: Whether to enable text matcher metrics
        workers: Number of parallel retriever workers, when applicable
        rag_optional_nodes_config: Effective optional RAG node settings, when overridden
    Returns:
        Dictionary containing the manifest payload for a validation run
    """
    rag_cfg = settings.rag_config
    optional_nodes = rag_optional_nodes_config or rag_cfg.optional_nodes.model_dump()
    rag_config: dict[str, Any] = {
        "embedding_model": rag_cfg.embedding.model_name,
        "reranker_model": rag_cfg.reranker.model_name,
        "chunk_size": rag_cfg.chunking.chunk_size,
        "chunk_overlap": rag_cfg.chunking.chunk_overlap,
        "embedding_query_instruction": rag_cfg.embedding.use_query_instruction,
        "reranker_query_instruction": rag_cfg.reranker.use_query_instruction,
        "retrieval": {
            "search_mode": str(rag_cfg.retrieval.search_mode),
            "fusion_method": str(rag_cfg.retrieval.fusion_method),
            "sparse_top_k": rag_cfg.retrieval.sparse_top_k,
            "dense_top_k": rag_cfg.retrieval.dense_top_k,
            "hybrid_top_k": rag_cfg.retrieval.hybrid_top_k,
        },
        "qdrant_collection": rag_cfg.qdrant_config.collection_name,
        "optional_nodes": optional_nodes,
    }
    if mode == "full":
        rag_config["llm_model"] = rag_cfg.llm.model_name

    run_parameters: dict[str, Any] = {
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "k": k,
        "enable_text_matcher_metrics": enable_text_matcher_metrics,
    }
    if workers is not None:
        run_parameters["workers"] = workers

    return {
        "mode": mode,
        "run_id": run_id,
        "timestamp_utc": timestamp_iso,
        "paths": {
            "output_dir": str(output_dir),
            "run_dir": str(run_dir),
            "dataset_file": str(dataset_file),
        },
        "run_parameters": run_parameters,
        "rag_config": rag_config,
        "rag_graph_profile": build_rag_graph_profile(
            mode=mode,
            optional_nodes_config=optional_nodes,
        ),
        "evaluation_config": {
            "ragas_metrics": ragas_metric_names,
            "gigachat_min_interval_sec": metrics_settings.GIGACHAT_MIN_INTERVAL_SEC,
            "gigachat_embeddings_max_text_chars": (
                metrics_settings.GIGACHAT_EMBEDDINGS_MAX_TEXT_CHARS
            ),
            "gigachat_embeddings_max_batch_chars": (
                metrics_settings.GIGACHAT_EMBEDDINGS_MAX_BATCH_CHARS
            ),
        },
    }


def build_rag_graph_profile(
    *,
    mode: str,
    optional_nodes_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a compact optional-node profile for validation artifacts.

    Args:
        mode: Evaluation mode (full or retriever)
        optional_nodes_config: Effective optional-node settings
    Returns:
        Dictionary with configured flags and mode-specific active optional nodes
    """
    configured = {
        field_name: optional_nodes_config.get(field_name) for field_name in RAG_GRAPH_PROFILE_FIELDS
    }
    effective_optional_nodes: list[str] = []
    inactive_configured_nodes: list[str] = []

    if bool(configured["enable_input_guardrails"]):
        effective_optional_nodes.append("input_guardrails")

    query_rewriter_enabled = bool(configured["enable_query_rewriter"])
    if query_rewriter_enabled:
        effective_optional_nodes.append("query_rewriter")

    if bool(configured["enable_query_clarification"]):
        if query_rewriter_enabled:
            effective_optional_nodes.append("query_clarification")
        else:
            inactive_configured_nodes.append("query_clarification")

    if bool(configured["enable_context_relevance_grader"]):
        effective_optional_nodes.append("context_relevance_grader")

    if bool(configured["enable_answer_formatter"]):
        if mode == "full":
            effective_optional_nodes.append("answer_formatter")
        else:
            inactive_configured_nodes.append("answer_formatter")

    if bool(configured["enable_output_guardrails"]):
        if mode == "full":
            effective_optional_nodes.append("output_guardrails")
        else:
            inactive_configured_nodes.append("output_guardrails")

    return {
        "mode": mode,
        "configured": configured,
        "effective_optional_nodes": effective_optional_nodes,
        "inactive_configured_nodes": inactive_configured_nodes,
    }


def _normalize_json_value(value: Any) -> Any:
    """
    Convert non-JSON-native values to serializable ones

    Args:
        value: Value to normalize
    Returns:
        Normalized value
    """
    normalized: Any
    if isinstance(value, dict):
        normalized = {str(k): _normalize_json_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        normalized = [_normalize_json_value(item) for item in value]
    elif isinstance(value, set):
        normalized = sorted(_normalize_json_value(item) for item in value)
    elif isinstance(value, Path):
        normalized = str(value)
    elif isinstance(value, Enum):
        normalized = value.value
    elif hasattr(value, "model_dump"):
        normalized = _normalize_json_value(value.model_dump())
    elif isinstance(value, (str, int, float, bool)) or value is None:
        normalized = value
    else:
        normalized = str(value)
    return normalized
