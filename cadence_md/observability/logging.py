"""Structured logging context and formatters."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from cadence_md.observability.privacy import redact_mapping
from cadence_md.observability.settings import ObservabilitySettings, observability_settings

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)
user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)
rag_request_id_ctx: ContextVar[str | None] = ContextVar("rag_request_id", default=None)
task_id_ctx: ContextVar[str | None] = ContextVar("task_id", default=None)

_STANDARD_RECORD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class ObservabilityContextFilter(logging.Filter):
    """Attach request/task contextvars to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = {
            "request_id": request_id_ctx.get() or "-",
            "correlation_id": correlation_id_ctx.get() or "-",
            "user_id": user_id_ctx.get() or "-",
            "rag_request_id": rag_request_id_ctx.get() or "-",
            "task_id": task_id_ctx.get() or "-",
        }
        for field, value in context.items():
            setattr(record, field, value)
        return True


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter that redacts known sensitive ``extra`` values."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
        }
        payload.update(redact_mapping(extras))
        return json.dumps(payload, ensure_ascii=False, default=str)


def current_log_context() -> dict[str, str]:
    """Return non-empty observability context for the current execution path."""
    raw = {
        "request_id": request_id_ctx.get(),
        "correlation_id": correlation_id_ctx.get(),
        "user_id": user_id_ctx.get(),
        "rag_request_id": rag_request_id_ctx.get(),
        "task_id": task_id_ctx.get(),
    }
    return {key: value for key, value in raw.items() if value}


@contextmanager
def log_context(
    *,
    request_id: str | None = None,
    correlation_id: str | None = None,
    user_id: str | None = None,
    rag_request_id: str | None = None,
    task_id: str | None = None,
) -> Iterator[None]:
    """Temporarily bind structured logging context fields."""
    tokens = []
    if request_id is not None:
        tokens.append((request_id_ctx, request_id_ctx.set(request_id)))
    if correlation_id is not None:
        tokens.append((correlation_id_ctx, correlation_id_ctx.set(correlation_id)))
    if user_id is not None:
        tokens.append((user_id_ctx, user_id_ctx.set(user_id)))
    if rag_request_id is not None:
        tokens.append((rag_request_id_ctx, rag_request_id_ctx.set(rag_request_id)))
    if task_id is not None:
        tokens.append((task_id_ctx, task_id_ctx.set(task_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def set_user_id(user_id: str) -> None:
    """Bind authenticated user id for the rest of the current request context."""
    user_id_ctx.set(user_id)


def configure_logging(settings: ObservabilitySettings | None = None) -> None:
    """Configure root logging with context enrichment and optional JSON output."""
    resolved = settings or observability_settings
    root = logging.getLogger()
    root.setLevel(resolved.LOG_LEVEL.upper())

    handler = logging.StreamHandler(sys.stdout)
    if resolved.LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] "
                "request_id=%(request_id)s rag_request_id=%(rag_request_id)s "
                "task_id=%(task_id)s %(message)s"
            )
        )
    handler.addFilter(ObservabilityContextFilter())

    root.handlers.clear()
    root.addHandler(handler)
