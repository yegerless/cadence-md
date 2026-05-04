"""
Centralized exponential backoff for transient inference and HTTP failures.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class RetryableHTTPStatusError(RuntimeError):
    """Carries HTTP status and a short body preview for :func:`is_retryable_error` / logging."""

    def __init__(self, status_code: int, *, body_preview: str = "") -> None:
        self.status_code = status_code
        self.body_preview = body_preview
        super().__init__(f"retryable HTTP {status_code}: {body_preview!r}")


def sleep_with_backoff(
    attempt: int,
    *,
    base_seconds: float = 1.0,
    max_seconds: float = 120.0,
    jitter_ratio: float = 0.1,
) -> None:
    """Sleep with capped exponential backoff and optional jitter.

    Args:
        attempt: Zero-based attempt index (0 after first failure).
        base_seconds: Base delay before exponentiation.
        max_seconds: Maximum sleep duration.
        jitter_ratio: Fraction of delay added as uniform random jitter.
    """
    delay = min(max_seconds, base_seconds * (2.0**attempt))
    jitter = delay * jitter_ratio * random.random()
    time.sleep(delay + jitter)


def is_retryable_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is commonly transient for inference clients."""
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(exc, APIStatusError) and status_code in RETRYABLE_HTTP_STATUS_CODES:
        return True

    if isinstance(exc, RetryableHTTPStatusError):
        return exc.status_code in RETRYABLE_HTTP_STATUS_CODES

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_HTTP_STATUS_CODES

    if isinstance(exc, httpx.RequestError):
        return True

    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def is_openai_retryable(exc: BaseException) -> bool:
    """Backward-compatible alias for older call sites/tests."""
    return is_retryable_error(exc)


def retry_sync[T](
    op: Callable[[], T],
    *,
    logger_: logging.Logger,
    max_attempts: int,
    operation_name: str,
    base_seconds: float = 1.0,
    max_seconds: float = 120.0,
    is_retryable: Callable[[BaseException], bool] | None = None,
) -> T:
    """Run ``op`` with retries on transient failures.

    Args:
        op: Zero-argument callable to execute.
        logger_: Logger for warnings on retries.
        max_attempts: Total attempts including the first.
        operation_name: Label for log messages.
        base_seconds: Backoff base delay.
        max_seconds: Backoff cap.
        is_retryable: Optional predicate overriding the default transient-error classifier.

    Returns:
        Result of ``op``.

    Raises:
        The last exception if all attempts fail.
    """
    max_attempts = max(1, max_attempts)
    retryable = is_retryable or is_retryable_error
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return op()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts - 1 or not retryable(exc):
                raise
            logger_.warning(
                "%s failed (%s); retrying attempt %s/%s",
                operation_name,
                exc,
                attempt + 2,
                max_attempts,
                exc_info=True,
            )
            sleep_with_backoff(attempt, base_seconds=base_seconds, max_seconds=max_seconds)
    assert last_exc is not None
    raise last_exc
