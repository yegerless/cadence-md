"""Prometheus metrics shared by backend, worker, and RAG pipeline."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from redis import Redis
from redis.exceptions import RedisError

from cadence_md.observability.settings import observability_settings

logger = logging.getLogger(__name__)

BACKEND_REQUESTS = Counter(
    "cadence_backend_requests",
    "FastAPI request count.",
    ("method", "path", "status"),
)
BACKEND_REQUEST_DURATION = Histogram(
    "cadence_backend_request_duration_seconds",
    "FastAPI request latency.",
    ("method", "path"),
)
BACKEND_REQUEST_ERRORS = Counter(
    "cadence_backend_request_errors",
    "FastAPI request errors.",
    ("method", "path", "error_type"),
)
RAG_NODE_DURATION = Histogram(
    "cadence_rag_node_duration_seconds",
    "RAG node latency.",
    ("node",),
)
RAG_FALLBACKS = Counter(
    "cadence_rag_fallbacks",
    "RAG fallback and truncation events.",
    ("type",),
)
CELERY_TASKS = Counter(
    "cadence_celery_tasks",
    "Celery task count.",
    ("task", "status"),
)
CELERY_TASK_DURATION = Histogram(
    "cadence_celery_task_duration_seconds",
    "Celery task latency.",
    ("task",),
)
CELERY_QUEUE_DEPTH = Gauge(
    "cadence_celery_queue_depth",
    "Approximate Celery broker queue depth from Redis LLEN.",
    ("queue",),
)


def metrics_enabled() -> bool:
    """Return whether Prometheus instrumentation is enabled."""
    return observability_settings.PROMETHEUS_ENABLED


def observe_backend_request(
    *,
    method: str,
    path: str,
    status_code: int,
    duration_seconds: float,
    error_type: str | None = None,
) -> None:
    """Record one backend HTTP request."""
    if not metrics_enabled():
        return
    status = str(status_code)
    BACKEND_REQUESTS.labels(method=method, path=path, status=status).inc()
    BACKEND_REQUEST_DURATION.labels(method=method, path=path).observe(duration_seconds)
    if error_type or status_code >= 500:
        BACKEND_REQUEST_ERRORS.labels(
            method=method,
            path=path,
            error_type=error_type or "HTTPError",
        ).inc()


def observe_rag_node(node: str, latency_ms: float | None) -> None:
    """Record RAG node latency in seconds."""
    if metrics_enabled() and latency_ms is not None:
        RAG_NODE_DURATION.labels(node=node).observe(latency_ms / 1000)


def inc_rag_fallback(fallback_type: str) -> None:
    """Increment a RAG fallback/truncation counter."""
    if metrics_enabled():
        RAG_FALLBACKS.labels(type=fallback_type).inc()


@contextmanager
def celery_task_timer(task_name: str) -> Iterator[None]:
    """Record Celery task duration and terminal status."""
    if not metrics_enabled():
        yield
        return
    started = time.perf_counter()
    status = "succeeded"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        CELERY_TASKS.labels(task=task_name, status=status).inc()
        CELERY_TASK_DURATION.labels(task=task_name).observe(time.perf_counter() - started)


def record_celery_task_result(task_name: str, status: str, duration_seconds: float) -> None:
    """Record an explicit Celery task result for async task internals and tests."""
    if not metrics_enabled():
        return
    CELERY_TASKS.labels(task=task_name, status=status).inc()
    CELERY_TASK_DURATION.labels(task=task_name).observe(duration_seconds)


def update_queue_depth(*, broker_url: str, queue_name: str) -> None:
    """Update the Redis LLEN queue-depth proxy gauge."""
    if not metrics_enabled():
        return
    try:
        depth = Redis.from_url(broker_url).llen(queue_name)
    except RedisError as exc:
        logger.warning(
            "Failed to collect Celery queue depth",
            extra={"queue": queue_name, "error_type": type(exc).__name__},
        )
        return
    CELERY_QUEUE_DEPTH.labels(queue=queue_name).set(float(depth))


def prometheus_payload(*, multiprocess_mode: bool = False) -> bytes:
    """Return Prometheus exposition bytes for the current process or multiprocess registry."""
    if multiprocess_mode:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest(REGISTRY)


def multiprocess_enabled() -> bool:
    """Return whether prometheus-client multiprocess mode appears configured."""
    configured = observability_settings.PROMETHEUS_MULTIPROC_DIR or os.environ.get(
        "PROMETHEUS_MULTIPROC_DIR"
    )
    return bool(configured)
