import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from cadence_md.app.settings import settings
from metrics.config import metrics_settings


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
    enable_text_matcher_metrics: bool = False,
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
        enable_text_matcher_metrics: Whether to enable text matcher metrics
    Returns:
        Dictionary containing the manifest payload for a validation run
    """
    rag_cfg = settings.rag_config
    return {
        "mode": mode,
        "run_id": run_id,
        "timestamp_utc": timestamp_iso,
        "paths": {
            "output_dir": str(output_dir),
            "run_dir": str(run_dir),
            "dataset_file": str(dataset_file),
        },
        "run_parameters": {
            "sample_size": sample_size,
            "k": k,
            "enable_text_matcher_metrics": enable_text_matcher_metrics,
        },
        "rag_config": {
            "llm_model": rag_cfg.llm.model_name,
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
        },
        "evaluation_config": {
            "ragas_metrics": ragas_metric_names,
            "gigachat_min_interval_sec": metrics_settings.GIGACHAT_MIN_INTERVAL_SEC,
        },
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
