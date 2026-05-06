"""Tests for Langfuse tracing adapter behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

from cadence_md.observability.langfuse import (
    SDKLangfuseTracer,
    TraceHandle,
    active_trace,
    record_span,
)
from cadence_md.observability.settings import ObservabilitySettings
from cadence_md.rag.contracts import RAGRequest
from cadence_md.rag.service import RAGService


class FakeTrace:
    """Minimal trace test double."""

    id = "trace-1"

    def __init__(self) -> None:
        self.spans: list[dict[str, object]] = []

    def span(self, **kwargs: object) -> FakeTrace:
        self.spans.append(kwargs)
        return self

    def end(self) -> None:
        return None


class FakeClient:
    """Minimal Langfuse client test double."""

    def __init__(self) -> None:
        self.trace_obj = FakeTrace()
        self.calls: list[dict[str, object]] = []

    def trace(self, **kwargs: object) -> FakeTrace:
        self.calls.append(kwargs)
        return self.trace_obj


def test_langfuse_tracer_redacts_query_by_default() -> None:
    client = FakeClient()
    tracer = SDKLangfuseTracer(
        client=client,
        settings=ObservabilitySettings(LANGFUSE_ENABLED=True, LANGFUSE_TRACE_QUERY_MODE="redacted"),
    )

    trace = tracer.start_trace(
        query="Полный медицинский вопрос",
        query_hash="hash",
        rag_request_id="rag-1",
        user_id="user-1",
        prompt_version="v1",
    )

    assert trace.trace_id == "trace-1"
    assert client.calls[0]["input"] == "[redacted medical query: hash]"
    assert client.calls[0]["user_id"] == "user-1"


def test_langfuse_tracer_failure_is_best_effort() -> None:
    client = MagicMock()
    client.trace.side_effect = RuntimeError("transport")
    tracer = SDKLangfuseTracer(
        client=client,
        settings=ObservabilitySettings(LANGFUSE_ENABLED=True),
    )

    trace = tracer.start_trace(
        query="q",
        query_hash="hash",
        rag_request_id=None,
        user_id=None,
        prompt_version="v1",
    )

    assert trace.trace_id is None


def test_active_trace_records_span_on_current_handle() -> None:
    trace_obj = FakeTrace()
    handle = TraceHandle(trace_id="trace-1", trace=trace_obj)

    with active_trace(handle):
        record_span("retrieve", metadata={"query_hash": "hash"})

    assert trace_obj.spans == [{"name": "retrieve", "metadata": {"query_hash": "hash"}}]


def test_rag_service_returns_langfuse_trace_id() -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = {
        "query": "q",
        "query_hash": "hash",
        "ranked_docs": [],
        "rerank_fallback": False,
        "retrieval_failed": False,
        "generate_fallback": False,
        "context_truncated": False,
        "error_type": None,
        "error_message": None,
        "sources": [],
        "context": "",
        "context_chars": 0,
        "answer": "answer",
        "answer_word_count": 1,
        "latency_ms": {},
    }
    tracer = MagicMock()
    tracer.start_trace.return_value = TraceHandle(trace_id="trace-1")

    response = RAGService(pipeline=pipeline, tracer=tracer).run(RAGRequest(query="q"))

    assert response.langfuse_trace_id == "trace-1"
