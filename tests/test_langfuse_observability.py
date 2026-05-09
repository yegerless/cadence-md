"""Tests for Langfuse tracing adapter behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

from cadence_md.observability.langfuse import (
    SDKLangfuseTracer,
    TraceHandle,
    active_trace,
    record_generation,
    record_span,
    record_trace_summary,
)
from cadence_md.observability.settings import ObservabilitySettings, observability_settings
from cadence_md.rag.contracts import RAGRequest
from cadence_md.rag.service import RAGService


class FakeObservation:
    """Minimal Langfuse v4 observation test double."""

    def __init__(self, trace_id: str = "trace-1") -> None:
        self.trace_id = trace_id
        self.children: list[FakeObservation] = []
        self.kwargs: dict[str, object] = {}
        self.updates: list[dict[str, object]] = []
        self.ended = False

    def start_observation(self, **kwargs: object) -> FakeObservation:
        child = FakeObservation(trace_id=self.trace_id)
        child.kwargs = dict(kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: object) -> None:
        self.updates.append(dict(kwargs))

    def end(self) -> None:
        self.ended = True


class FakeObservationContext:
    """Context manager returned by the Langfuse v4 client."""

    def __init__(self, client: FakeClient, kwargs: dict[str, object]) -> None:
        self.client = client
        self.kwargs = kwargs
        self.observation = FakeObservation()
        self.exited = False

    def __enter__(self) -> FakeObservation:
        self.client.active_observation = self.observation
        return self.observation

    def __exit__(self, *_: object) -> None:
        self.exited = True
        self.observation.end()
        self.client.active_observation = None


class FakeAttributesContext:
    """Context manager returned by ``propagate_attributes``."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self) -> FakeAttributesContext:
        self.entered = True
        return self

    def __exit__(self, *_: object) -> None:
        self.exited = True


class FakeClient:
    """Minimal Langfuse v4 client test double."""

    def __init__(self) -> None:
        self.contexts: list[FakeObservationContext] = []
        self.calls: list[dict[str, object]] = []
        self.flush_calls = 0
        self.active_observation: FakeObservation | None = None

    def start_as_current_observation(self, **kwargs: object) -> FakeObservationContext:
        self.calls.append(dict(kwargs))
        context = FakeObservationContext(self, dict(kwargs))
        self.contexts.append(context)
        return context

    def get_current_trace_id(self) -> str | None:
        if self.active_observation is None:
            return None
        return self.active_observation.trace_id

    def flush(self) -> None:
        self.flush_calls += 1


def _make_tracer(
    client: FakeClient,
    *,
    settings: ObservabilitySettings | None = None,
    attributes_context: FakeAttributesContext | None = None,
) -> SDKLangfuseTracer:
    tracer = SDKLangfuseTracer(
        client=client,
        settings=settings or ObservabilitySettings(LANGFUSE_ENABLED=True),
    )
    tracer._attributes_context = MagicMock(
        return_value=attributes_context or FakeAttributesContext()
    )
    return tracer


def test_langfuse_tracer_redacts_query_by_default() -> None:
    client = FakeClient()
    tracer = _make_tracer(
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
    with active_trace(trace):
        pass

    assert trace.trace_id == "trace-1"
    assert client.calls[0]["input"] == "[redacted medical query: hash]"
    assert client.calls[0]["version"] == client.calls[0]["metadata"]["prompt_version"]
    assert "Полный медицинский вопрос" not in str(client.calls[0]["metadata"])
    assert client.contexts[0].exited is True
    assert client.flush_calls == 1


def test_langfuse_tracer_hash_mode_does_not_store_raw_query_in_metadata() -> None:
    client = FakeClient()
    tracer = _make_tracer(
        client=client,
        settings=ObservabilitySettings(LANGFUSE_ENABLED=True, LANGFUSE_TRACE_QUERY_MODE="hash"),
    )

    trace = tracer.start_trace(
        query="Пациент с гипертензией и диабетом",
        query_hash="hash-only",
        rag_request_id="rag-1",
        user_id="user-1",
        prompt_version="v1",
    )
    with active_trace(trace):
        pass

    assert client.calls[0]["input"] == "hash-only"
    assert "гипертензией" not in str(client.calls[0]["metadata"])


def test_langfuse_tracer_failure_is_best_effort() -> None:
    client = MagicMock()
    client.start_as_current_observation.side_effect = RuntimeError("transport")
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


def test_langfuse_tracer_warns_when_v4_api_is_unavailable(caplog) -> None:
    client = object()
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
    assert "Langfuse SDK v4 observations API unavailable" in caplog.text


def test_langfuse_tracer_defaults_base_url_when_host_env_is_empty() -> None:
    tracer = SDKLangfuseTracer(
        settings=ObservabilitySettings(
            LANGFUSE_ENABLED=True,
            LANGFUSE_BASE_URL="",
            LANGFUSE_HOST="",
        )
    )

    assert tracer._resolve_base_url() == "https://cloud.langfuse.com"


def test_langfuse_tracer_prefers_base_url_over_deprecated_host() -> None:
    tracer = SDKLangfuseTracer(
        settings=ObservabilitySettings(
            LANGFUSE_ENABLED=True,
            LANGFUSE_BASE_URL="https://cloud.langfuse.com/",
            LANGFUSE_HOST="http://localhost:3000",
        )
    )

    assert tracer._resolve_base_url() == "https://cloud.langfuse.com"


def test_langfuse_tracer_normalizes_host_without_scheme() -> None:
    tracer = SDKLangfuseTracer(
        settings=ObservabilitySettings(
            LANGFUSE_ENABLED=True,
            LANGFUSE_BASE_URL=None,
            LANGFUSE_HOST="cloud.langfuse.com",
        )
    )

    assert tracer._resolve_base_url() == "https://cloud.langfuse.com"


def test_langfuse_tracer_rejects_invalid_base_url(caplog) -> None:
    tracer = SDKLangfuseTracer(
        settings=ObservabilitySettings(
            LANGFUSE_ENABLED=True,
            LANGFUSE_BASE_URL="/api/public/otel/v1/traces",
        )
    )

    assert tracer._resolve_base_url() is None
    assert "Langfuse base URL is invalid" in caplog.text


def test_active_trace_records_span_on_current_handle() -> None:
    client = FakeClient()
    handle = TraceHandle(
        client=client,
        root_context=client.start_as_current_observation(name="rag_request", as_type="span"),
    )

    with active_trace(handle):
        record_span("retrieve", metadata={"query_hash": "hash"})

    root = client.contexts[0].observation
    assert root.children[0].kwargs == {
        "name": "retrieve",
        "as_type": "span",
        "metadata": {"query_hash": "hash"},
    }
    assert root.children[0].ended is True
    assert root.ended is True
    assert client.flush_calls == 1


def test_active_trace_records_span_input_and_output_on_current_handle() -> None:
    client = FakeClient()
    handle = TraceHandle(
        client=client,
        root_context=client.start_as_current_observation(name="rag_request", as_type="span"),
    )

    with active_trace(handle):
        record_span(
            "retrieve",
            input_data={"query": "[redacted query]"},
            output_data={"n_docs": 2},
            metadata={"query_hash": "hash"},
        )

    span = client.contexts[0].observation.children[0]
    assert span.kwargs == {
        "name": "retrieve",
        "as_type": "span",
        "metadata": {"query_hash": "hash"},
        "input": {"query": "[redacted query]"},
        "output": {"n_docs": 2},
    }
    assert span.updates == [{"output": {"n_docs": 2}, "metadata": {"query_hash": "hash"}}]


def test_active_trace_records_generation_on_current_handle() -> None:
    client = FakeClient()
    handle = TraceHandle(
        client=client,
        root_context=client.start_as_current_observation(name="rag_request", as_type="span"),
    )

    with active_trace(handle):
        record_generation(
            "generate",
            input_data="[redacted query]",
            output_data="[redacted answer]",
            metadata={"query_hash": "hash"},
        )

    generation = client.contexts[0].observation.children[0]
    assert generation.kwargs == {
        "name": "generate",
        "as_type": "generation",
        "metadata": {"query_hash": "hash"},
        "input": "[redacted query]",
        "output": "[redacted answer]",
    }
    assert generation.updates == [
        {"output": "[redacted answer]", "metadata": {"query_hash": "hash"}}
    ]
    assert generation.ended is True


def test_active_trace_records_generation_model_and_version() -> None:
    client = FakeClient()
    handle = TraceHandle(
        client=client,
        root_context=client.start_as_current_observation(name="rag_request", as_type="span"),
    )

    with active_trace(handle):
        record_generation(
            "generate",
            input_data="[redacted query]",
            output_data="[redacted answer]",
            metadata={"query_hash": "hash", "model": "model-1", "prompt_version": "prompt-v1"},
        )

    generation = client.contexts[0].observation.children[0]
    assert generation.kwargs["model"] == "model-1"
    assert generation.kwargs["version"] == "prompt-v1"


def test_active_trace_updates_root_summary() -> None:
    client = FakeClient()
    handle = TraceHandle(
        client=client,
        root_context=client.start_as_current_observation(name="rag_request", as_type="span"),
    )

    with active_trace(handle):
        record_trace_summary(
            output_data="[redacted rag answer]",
            metadata={"latency_ms": {"llm": 12.3}, "flags": {"generate_fallback": False}},
        )

    root = client.contexts[0].observation
    assert root.updates == [
        {
            "output": "[redacted rag answer]",
            "metadata": {
                "latency_ms": {"llm": 12.3},
                "flags": {"generate_fallback": False},
            },
        }
    ]


def test_active_trace_enters_attribute_context() -> None:
    client = FakeClient()
    attributes_context = FakeAttributesContext()
    tracer = _make_tracer(client, attributes_context=attributes_context)
    trace = tracer.start_trace(
        query="q",
        query_hash="hash",
        rag_request_id="rag-1",
        user_id="user-1",
        prompt_version="v1",
    )

    with active_trace(trace):
        assert attributes_context.entered is True

    assert attributes_context.exited is True


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


def test_rag_service_updates_langfuse_root_summary(monkeypatch) -> None:
    monkeypatch.setattr(observability_settings, "LANGFUSE_TRACE_QUERY_MODE", "redacted")
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
        "context_chars": 42,
        "answer": "answer",
        "answer_word_count": 1,
        "latency_ms": {"llm": 12.3, "qdrant": 4.5},
    }
    trace = TraceHandle(trace_id="trace-1")
    trace.update = MagicMock()
    tracer = MagicMock()
    tracer.start_trace.return_value = trace

    RAGService(pipeline=pipeline, tracer=tracer).run(RAGRequest(query="q"))

    trace.update.assert_called_once()
    _, kwargs = trace.update.call_args
    assert kwargs["output_data"] == "[redacted rag answer]"
    assert kwargs["metadata"]["query_hash"] == "hash"
    assert kwargs["metadata"]["latency_ms"] == {"qdrant": 4.5, "llm": 12.3, "total_ms": 16.8}
    assert kwargs["metadata"]["flags"]["generate_fallback"] is False
    assert kwargs["metadata"]["source_count"] == 0
    assert kwargs["metadata"]["context_chars"] == 42
