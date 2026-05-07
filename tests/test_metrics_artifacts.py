"""Tests for metrics artifact helpers (JSONL, run dirs, manifest)."""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest
from langchain_core.documents import Document
from pydantic import BaseModel

import metrics.artifacts as artifacts_mod
from metrics.artifacts import (
    _normalize_json_value,
    append_jsonl,
    build_run_directory,
    build_run_id,
    create_run_manifest,
    document_to_record,
    now_utc_iso,
    write_json,
)


def test_now_utc_iso_is_valid_utc_iso() -> None:
    ts = now_utc_iso()
    normalized = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    parsed = datetime.fromisoformat(normalized)
    assert parsed.tzinfo is not None


def test_build_run_id_strips_separators_and_timezone() -> None:
    rid = build_run_id("full", "2026-05-03T12:00:00+00:00")
    assert rid == "full_20260503T120000Z"


def test_build_run_directory_creates_unique_dir(tmp_path: Path) -> None:
    run_dir, run_id, ts = build_run_directory(tmp_path, mode="retriever")
    assert run_dir.exists()
    assert run_dir.name == run_id
    assert run_id.startswith("retriever_")
    assert len(ts) > 10


def test_build_run_directory_raises_if_same_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_ts = "2026-05-03T12:00:00+00:00"
    monkeypatch.setattr(artifacts_mod, "now_utc_iso", lambda: frozen_ts)

    run_dir1, run_id1, _ = build_run_directory(tmp_path, mode="full")
    assert run_dir1.exists()
    assert run_id1 == build_run_id("full", frozen_ts)

    with pytest.raises(FileExistsError):
        build_run_directory(tmp_path, mode="full")


def test_append_jsonl_utf8_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "out" / "rows.jsonl"
    append_jsonl(path, {"msg": "кириллица", "n": 1})
    append_jsonl(path, {"msg": "ещё", "n": 2})
    text = path.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    assert len(lines) == 2
    assert "кириллица" in lines[0]
    assert json.loads(lines[0])["msg"] == "кириллица"


def test_write_json_pretty_and_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_json(path, {"a": 1})
    first = path.read_text(encoding="utf-8")
    assert "\n" in first
    write_json(path, {"b": 2})
    second = path.read_text(encoding="utf-8")
    assert json.loads(second) == {"b": 2}


def test_document_to_record_with_and_without_score() -> None:
    doc = Document(page_content="hello", metadata={"section_id": "s1"})
    assert document_to_record(doc) == {
        "page_content": "hello",
        "metadata": {"section_id": "s1"},
    }
    assert document_to_record(doc, score=0.42) == {
        "page_content": "hello",
        "metadata": {"section_id": "s1"},
        "score": 0.42,
    }


class _Color(Enum):
    RED = "red"


class _TinyModel(BaseModel):
    x: int


def test_normalize_json_value_variants() -> None:
    path = Path("artifacts") / "run.json"
    assert _normalize_json_value(path) == str(path)
    assert _normalize_json_value(_Color.RED) == "red"
    assert _normalize_json_value(_TinyModel(x=3)) == {"x": 3}
    assert _normalize_json_value((1, 2)) == [1, 2]
    assert _normalize_json_value({3, 1, 2}) == [1, 2, 3]
    assert _normalize_json_value({1: "a", "b": 2}) == {"1": "a", "b": 2}

    class _NoDump:
        def __str__(self) -> str:
            return "nodump"

    assert _normalize_json_value(_NoDump()) == "nodump"


def test_create_run_manifest_shape() -> None:
    out = Path("/tmp/metrics_out")
    run_dir = out / "full_20260101"
    manifest = create_run_manifest(
        mode="full",
        run_id="full_20260101",
        timestamp_iso="2026-01-01T00:00:00+00:00",
        output_dir=out,
        run_dir=run_dir,
        dataset_file=Path("data/qa.jsonl"),
        sample_size=10,
        k=5,
        ragas_metric_names=["faithfulness"],
        enable_text_matcher_metrics=True,
    )
    assert manifest["mode"] == "full"
    assert manifest["run_id"] == "full_20260101"
    assert manifest["paths"]["dataset_file"] == "data/qa.jsonl"
    assert manifest["run_parameters"]["sample_size"] == 10
    assert manifest["run_parameters"]["k"] == 5
    assert manifest["run_parameters"]["enable_text_matcher_metrics"] is True
    assert manifest["evaluation_config"]["ragas_metrics"] == ["faithfulness"]
    assert "gigachat_min_interval_sec" in manifest["evaluation_config"]
    assert "llm_model" in manifest["rag_config"]
    assert "retrieval" in manifest["rag_config"]
    assert manifest["rag_graph_profile"]["mode"] == "full"
    assert manifest["rag_graph_profile"]["configured"]["max_query_rewrite_iterations"] == 1
    assert "answer_formatter" in manifest["rag_graph_profile"]["effective_optional_nodes"]


def test_create_run_manifest_retriever_excludes_llm_model() -> None:
    out = Path("/tmp/metrics_out")
    run_dir = out / "retriever_20260101"
    manifest = create_run_manifest(
        mode="retriever",
        run_id="retriever_20260101",
        timestamp_iso="2026-01-01T00:00:00+00:00",
        output_dir=out,
        run_dir=run_dir,
        dataset_file=Path("data/qa.jsonl"),
        sample_size=10,
        k=5,
        ragas_metric_names=[],
        enable_text_matcher_metrics=False,
    )
    assert manifest["mode"] == "retriever"
    assert "llm_model" not in manifest["rag_config"]
    assert "retrieval" in manifest["rag_config"]
    assert manifest["rag_graph_profile"]["mode"] == "retriever"
    assert "answer_formatter" in manifest["rag_graph_profile"]["inactive_configured_nodes"]
