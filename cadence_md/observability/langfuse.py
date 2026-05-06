"""Best-effort Langfuse tracing for RAG requests."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

from cadence_md.observability.privacy import redact_medical_query
from cadence_md.observability.settings import ObservabilitySettings, observability_settings

logger = logging.getLogger(__name__)


class LangfuseTracer(Protocol):
    """Small protocol used by RAGService without binding it to SDK details."""

    def start_trace(
        self,
        *,
        query: str,
        query_hash: str,
        rag_request_id: str | None,
        user_id: str | None,
        prompt_version: str,
    ) -> TraceHandle:
        """Create a trace handle or no-op handle."""


@dataclass
class TraceHandle:
    """Active trace wrapper with a stable id and best-effort child event helpers."""

    trace_id: str | None = None
    trace: Any | None = None

    def span(self, name: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Create an instantaneous span/event if the SDK object supports it."""
        if self.trace is None:
            return
        try:
            span_factory = getattr(self.trace, "span", None)
            if callable(span_factory):
                span = span_factory(name=name, metadata=metadata or {})
                end = getattr(span, "end", None)
                if callable(end):
                    end()
        except Exception as exc:  # pragma: no cover - SDK-specific best effort
            logger.warning(
                "Langfuse span export failed",
                extra={"span": name, "error_type": type(exc).__name__},
            )

    def generation(
        self,
        name: str,
        *,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a generation observation if the SDK object supports it."""
        if self.trace is None:
            return
        try:
            generation_factory = getattr(self.trace, "generation", None)
            if callable(generation_factory):
                generation_factory(
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata=metadata or {},
                )
        except Exception as exc:  # pragma: no cover - SDK-specific best effort
            logger.warning(
                "Langfuse generation export failed",
                extra={"generation": name, "error_type": type(exc).__name__},
            )


class NoopLangfuseTracer:
    """Tracer used when Langfuse is disabled or unavailable."""

    def start_trace(
        self,
        *,
        query: str,
        query_hash: str,
        rag_request_id: str | None,
        user_id: str | None,
        prompt_version: str,
    ) -> TraceHandle:
        return TraceHandle()


class SDKLangfuseTracer:
    """Thin defensive adapter around the Langfuse Python SDK."""

    def __init__(
        self,
        *,
        settings: ObservabilitySettings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or observability_settings
        self._client = client

    @property
    def client(self) -> Any | None:
        """Lazily build the SDK client only when tracing is enabled."""
        if self._client is not None:
            return self._client
        if not self.settings.LANGFUSE_ENABLED:
            return None
        try:
            from langfuse import Langfuse  # noqa: PLC0415

            kwargs = {
                "public_key": self.settings.LANGFUSE_PUBLIC_KEY,
                "secret_key": self.settings.LANGFUSE_SECRET_KEY,
            }
            if self.settings.LANGFUSE_HOST:
                kwargs["host"] = self.settings.LANGFUSE_HOST
            self._client = Langfuse(**{key: value for key, value in kwargs.items() if value})
        except Exception as exc:
            logger.warning(
                "Langfuse client initialization failed",
                extra={"error_type": type(exc).__name__},
            )
            self._client = None
        return self._client

    def start_trace(
        self,
        *,
        query: str,
        query_hash: str,
        rag_request_id: str | None,
        user_id: str | None,
        prompt_version: str,
    ) -> TraceHandle:
        client = self.client
        if client is None:
            return TraceHandle()
        if self.settings.LANGFUSE_TRACE_QUERY_MODE == "full":
            input_data = redact_medical_query(query, "full")
        elif self.settings.LANGFUSE_TRACE_QUERY_MODE == "hash":
            input_data = query_hash
        else:
            input_data = f"[redacted medical query: {query_hash}]"
        metadata = {
            "query_hash": query_hash,
            "rag_request_id": rag_request_id,
            "prompt_version": self.settings.LANGFUSE_PROMPT_VERSION or prompt_version,
        }
        try:
            trace_factory = getattr(client, "trace", None)
            if callable(trace_factory):
                trace = trace_factory(
                    name="rag_request",
                    input=input_data,
                    user_id=user_id,
                    metadata=metadata,
                )
                trace_id = str(getattr(trace, "id", None) or getattr(trace, "trace_id", "") or "")
                return TraceHandle(trace_id=trace_id or None, trace=trace)
        except Exception as exc:
            logger.warning(
                "Langfuse trace creation failed",
                extra={"query_hash": query_hash, "error_type": type(exc).__name__},
            )
        return TraceHandle()


active_trace_ctx: ContextVar[TraceHandle | None] = ContextVar("langfuse_trace", default=None)


@contextmanager
def active_trace(trace: TraceHandle) -> Any:
    """Bind an active trace to RAG node execution."""
    token = active_trace_ctx.set(trace)
    try:
        yield trace
    finally:
        active_trace_ctx.reset(token)


def current_trace() -> TraceHandle | None:
    """Return the active Langfuse trace handle, if one exists."""
    return active_trace_ctx.get()


def record_span(name: str, *, metadata: dict[str, Any] | None = None) -> None:
    """Record a best-effort Langfuse span on the active trace."""
    trace = current_trace()
    if trace is not None:
        trace.span(name, metadata=metadata)


def record_generation(
    name: str,
    *,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a best-effort Langfuse generation on the active trace."""
    trace = current_trace()
    if trace is not None:
        trace.generation(name, input_data=input_data, output_data=output_data, metadata=metadata)


def get_langfuse_tracer() -> LangfuseTracer:
    """Return the configured tracer implementation."""
    if not observability_settings.LANGFUSE_ENABLED:
        return NoopLangfuseTracer()
    return SDKLangfuseTracer()
