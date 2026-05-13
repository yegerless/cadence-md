"""Prometheus HTTP endpoint for the RAG Celery worker."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from wsgiref.simple_server import make_server

from prometheus_client import CONTENT_TYPE_LATEST

from cadence_md.observability.metrics import multiprocess_enabled, prometheus_payload
from cadence_md.observability.settings import observability_settings

logger = logging.getLogger(__name__)

_server_thread: threading.Thread | None = None


def _metrics_app(
    environ: dict[str, object],
    start_response: Callable[[str, list[tuple[str, str]]], None],
) -> list[bytes]:
    path = str(environ.get("PATH_INFO") or "")
    if path != "/metrics":
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"not found"]
    payload = prometheus_payload(multiprocess_mode=multiprocess_enabled())
    start_response("200 OK", [("Content-Type", CONTENT_TYPE_LATEST)])
    return [payload]


def start_worker_metrics_server() -> None:
    """Start the worker metrics endpoint once per Celery worker container."""
    global _server_thread  # noqa: PLW0603
    if not observability_settings.PROMETHEUS_ENABLED or _server_thread is not None:
        return

    port = observability_settings.WORKER_METRICS_PORT

    def _serve() -> None:
        with make_server("0.0.0.0", port, _metrics_app) as httpd:
            logger.info("Worker metrics endpoint started", extra={"port": port})
            httpd.serve_forever()

    _server_thread = threading.Thread(target=_serve, name="worker-metrics", daemon=True)
    _server_thread.start()
