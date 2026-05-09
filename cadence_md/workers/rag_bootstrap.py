"""Worker-local RAG runtime bootstrap and cleanup."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadence_md.rag.service import RAGService

logger = logging.getLogger(__name__)


@dataclass
class RAGWorkerRuntime:
    """RAG service plus resources that need explicit worker shutdown."""

    service: RAGService
    close_hooks: list[Callable[[], None]] = field(default_factory=list)

    def warmup(self) -> None:
        """Run lightweight checks that do not call the LLM."""
        pipeline = self.service.pipeline
        ensure_compiled = getattr(pipeline, "ensure_compiled", None)
        if callable(ensure_compiled):
            ensure_compiled()
        logger.info("RAG worker runtime warmed up")

    def close(self) -> None:
        """Close all registered runtime resources."""
        for close_hook in self.close_hooks:
            try:
                close_hook()
            except Exception:
                logger.exception("Failed to close RAG worker runtime resource")


@dataclass
class _RuntimeState:
    """Mutable holder for the process-local runtime."""

    runtime: RAGWorkerRuntime | None = None


_runtime_state = _RuntimeState()
_runtime_lock = threading.Lock()


def build_rag_worker_runtime() -> RAGWorkerRuntime:
    """Build the heavy RAG stack for a worker process."""
    from cadence_md.rag.bootstrap import build_rag_stack  # noqa: PLC0415
    from cadence_md.rag.service import RAGService  # noqa: PLC0415

    pipeline, reranker = build_rag_stack()
    runtime = RAGWorkerRuntime(service=RAGService(pipeline), close_hooks=[reranker.close])
    runtime.warmup()
    return runtime


def get_rag_runtime() -> RAGWorkerRuntime:
    """Return a process-local cached RAG runtime."""
    if _runtime_state.runtime is not None:
        return _runtime_state.runtime

    with _runtime_lock:
        if _runtime_state.runtime is None:
            logger.info("Building RAG worker runtime")
            _runtime_state.runtime = build_rag_worker_runtime()
        return _runtime_state.runtime


def set_rag_runtime_for_tests(runtime: RAGWorkerRuntime | None) -> None:
    """Override the cached runtime in tests."""
    with _runtime_lock:
        _runtime_state.runtime = runtime


def shutdown_rag_runtime() -> None:
    """Close and clear the cached RAG runtime."""
    with _runtime_lock:
        runtime = _runtime_state.runtime
        _runtime_state.runtime = None

    if runtime is not None:
        runtime.close()
