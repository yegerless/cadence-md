"""Best-effort Langfuse tracing for RAG requests."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from cadence_md.observability.privacy import redact_medical_query
from cadence_md.observability.settings import ObservabilitySettings, observability_settings

logger = logging.getLogger(__name__)
DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com"


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
    client: Any | None = None
    root_context: Any | None = None
    attributes_context: Any | None = None
    trace: Any | None = None
    _entered_contexts: list[Any] = field(default_factory=list, init=False, repr=False)

    def enter(self) -> None:
        """Enter the root Langfuse observation context if tracing is enabled."""
        if self.root_context is None:
            return
        try:
            root_enter = getattr(self.root_context, "__enter__", None)
            self.trace = root_enter() if callable(root_enter) else self.root_context
            self._entered_contexts.append(self.root_context)

            if self.attributes_context is not None:
                attributes_enter = getattr(self.attributes_context, "__enter__", None)
                if callable(attributes_enter):
                    attributes_enter()
                    self._entered_contexts.append(self.attributes_context)

            self._capture_trace_id()
        except Exception as exc:  # pragma: no cover - SDK-specific best effort
            logger.warning(
                "Langfuse root observation start failed",
                extra={"error_type": type(exc).__name__},
            )
            self.exit()
            self.trace = None

    def exit(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: Any | None = None,
    ) -> None:
        """Exit the root observation context and flush buffered Langfuse events."""
        while self._entered_contexts:
            context = self._entered_contexts.pop()
            try:
                context_exit = getattr(context, "__exit__", None)
                if callable(context_exit):
                    context_exit(exc_type, exc, traceback)
            except Exception as close_exc:  # pragma: no cover - SDK-specific best effort
                logger.warning(
                    "Langfuse observation close failed",
                    extra={"error_type": type(close_exc).__name__},
                )

        if self.client is not None:
            try:
                flush = getattr(self.client, "flush", None)
                if callable(flush):
                    flush()
            except Exception as flush_exc:  # pragma: no cover - SDK-specific best effort
                logger.warning(
                    "Langfuse flush failed",
                    extra={"error_type": type(flush_exc).__name__},
                )

    def _capture_trace_id(self) -> None:
        """Populate ``trace_id`` from the active observation or SDK context."""
        if self.trace_id:
            return
        trace_id = getattr(self.trace, "trace_id", None)
        if not trace_id and self.client is not None:
            current_trace_id = getattr(self.client, "get_current_trace_id", None)
            if callable(current_trace_id):
                trace_id = current_trace_id()
        if trace_id:
            self.trace_id = str(trace_id)

    def _start_observation(
        self,
        name: str,
        *,
        as_type: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any | None:
        """Start a child observation under the active root observation."""
        if self.trace is None:
            return None

        start_observation = getattr(self.trace, "start_observation", None)
        if not callable(start_observation) and self.client is not None:
            start_observation = getattr(self.client, "start_observation", None)
        if not callable(start_observation):
            return None

        kwargs = {
            "name": name,
            "as_type": as_type,
            "metadata": metadata or {},
        }
        if input_data is not None:
            kwargs["input"] = input_data
        if output_data is not None:
            kwargs["output"] = output_data
        if metadata:
            model = metadata.get("model")
            if isinstance(model, str) and model:
                kwargs["model"] = model
            prompt_version = metadata.get("prompt_version")
            if isinstance(prompt_version, str) and prompt_version:
                kwargs["version"] = prompt_version
        return start_observation(**kwargs)

    def update(self, *, output_data: Any = None, metadata: dict[str, Any] | None = None) -> None:
        """Update the root observation with final RAG output and summary metadata."""
        if self.trace is None:
            return
        try:
            update = getattr(self.trace, "update", None)
            if callable(update):
                kwargs: dict[str, Any] = {}
                if output_data is not None:
                    kwargs["output"] = output_data
                if metadata is not None:
                    kwargs["metadata"] = metadata
                if kwargs:
                    update(**kwargs)
        except Exception as exc:  # pragma: no cover - SDK-specific best effort
            logger.warning(
                "Langfuse root observation update failed",
                extra={"error_type": type(exc).__name__},
            )

    def span(
        self,
        name: str,
        *,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create an instantaneous span/event if the SDK object supports it."""
        try:
            span = self._start_observation(
                name,
                as_type="span",
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
            )
            if span is None:
                return
            update = getattr(span, "update", None)
            if callable(update) and (output_data is not None or metadata):
                kwargs: dict[str, Any] = {}
                if output_data is not None:
                    kwargs["output"] = output_data
                if metadata:
                    kwargs["metadata"] = metadata
                update(**kwargs)
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
        try:
            generation = self._start_observation(
                name,
                as_type="generation",
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
            )
            if generation is None:
                return
            update = getattr(generation, "update", None)
            if callable(update) and (output_data is not None or metadata):
                kwargs: dict[str, Any] = {}
                if output_data is not None:
                    kwargs["output"] = output_data
                if metadata:
                    kwargs["metadata"] = metadata
                update(**kwargs)
            end = getattr(generation, "end", None)
            if callable(end):
                end()
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

            base_url = self._resolve_base_url()
            if base_url is None:
                return None
            kwargs = {
                "public_key": self.settings.LANGFUSE_PUBLIC_KEY,
                "secret_key": self.settings.LANGFUSE_SECRET_KEY,
                "base_url": base_url,
            }
            self._client = Langfuse(**{key: value for key, value in kwargs.items() if value})
        except Exception as exc:
            logger.warning(
                "Langfuse client initialization failed",
                extra={"error_type": type(exc).__name__},
            )
            self._client = None
        return self._client

    def _resolve_base_url(self) -> str | None:
        """Return an absolute Langfuse base URL, independent of empty env vars."""
        raw_url = self.settings.LANGFUSE_BASE_URL or self.settings.LANGFUSE_HOST
        base_url = (raw_url or DEFAULT_LANGFUSE_BASE_URL).strip().rstrip("/")
        if not base_url:
            base_url = DEFAULT_LANGFUSE_BASE_URL

        if "://" not in base_url:
            if base_url.startswith(("localhost", "127.0.0.1", "host.docker.internal")):
                base_url = f"http://{base_url}"
            else:
                base_url = f"https://{base_url}"

        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            logger.warning(
                "Langfuse base URL is invalid; tracing disabled",
                extra={"base_url": base_url},
            )
            return None
        return base_url

    def _attributes_context(
        self,
        *,
        user_id: str | None,
        metadata: dict[str, Any],
        prompt_version: str,
    ) -> Any | None:
        """Create a v4 attribute propagation context when the SDK exposes one."""
        try:
            from langfuse import propagate_attributes  # noqa: PLC0415
        except Exception as exc:
            logger.warning(
                "Langfuse attribute propagation unavailable",
                extra={"error_type": type(exc).__name__},
            )
            return None

        kwargs: dict[str, Any] = {
            "metadata": metadata,
            "trace_name": "rag_request",
            "version": prompt_version,
        }
        if user_id:
            kwargs["user_id"] = user_id
        try:
            return propagate_attributes(**kwargs)
        except Exception as exc:  # pragma: no cover - SDK-specific best effort
            logger.warning(
                "Langfuse attribute propagation setup failed",
                extra={"error_type": type(exc).__name__},
            )
            return None

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
        metadata = {key: value for key, value in metadata.items() if value is not None}
        try:
            observation_factory = getattr(client, "start_as_current_observation", None)
            if not callable(observation_factory):
                logger.warning(
                    "Langfuse SDK v4 observations API unavailable",
                    extra={"client_type": type(client).__name__},
                )
                return TraceHandle()

            root_context = observation_factory(
                as_type="span",
                name="rag_request",
                input=input_data,
                metadata=metadata,
                version=str(metadata["prompt_version"]),
            )
            attributes_context = self._attributes_context(
                user_id=user_id,
                metadata=metadata,
                prompt_version=str(metadata["prompt_version"]),
            )
            return TraceHandle(
                client=client,
                root_context=root_context,
                attributes_context=attributes_context,
            )
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
    trace.enter()
    token = active_trace_ctx.set(trace)
    try:
        yield trace
    except BaseException as exc:
        trace.exit(type(exc), exc, exc.__traceback__)
        raise
    else:
        trace.exit()
    finally:
        active_trace_ctx.reset(token)


def current_trace() -> TraceHandle | None:
    """Return the active Langfuse trace handle, if one exists."""
    return active_trace_ctx.get()


def record_span(
    name: str,
    *,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a best-effort Langfuse span on the active trace."""
    trace = current_trace()
    if trace is not None:
        trace.span(name, input_data=input_data, output_data=output_data, metadata=metadata)


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


def record_trace_summary(
    *,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Update the active root trace with final summary fields."""
    trace = current_trace()
    if trace is not None:
        trace.update(output_data=output_data, metadata=metadata)


def get_langfuse_tracer() -> LangfuseTracer:
    """Return the configured tracer implementation."""
    if not observability_settings.LANGFUSE_ENABLED:
        return NoopLangfuseTracer()
    return SDKLangfuseTracer()
