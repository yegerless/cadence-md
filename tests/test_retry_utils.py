"""Unit tests for cadence_md.app.retry_utils (predicates, backoff, retry_sync)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from unittest.mock import MagicMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from cadence_md.app.retry_utils import (
    RETRYABLE_HTTP_STATUS_CODES,
    RetryableHTTPStatusError,
    is_openai_retryable,
    is_retryable_error,
    retry_sync,
    sleep_with_backoff,
)


def _httpx_response(status_code: int) -> httpx.Response:
    req = httpx.Request("GET", "https://example.invalid/inference")
    return httpx.Response(status_code, request=req)


# --- is_retryable_error / is_openai_retryable ---


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: RateLimitError("too many", response=_httpx_response(429), body=None), True),
        (lambda: APIConnectionError(request=httpx.Request("GET", "https://x")), True),
        (lambda: APITimeoutError(request=httpx.Request("GET", "https://x")), True),
        (lambda: InternalServerError("x", response=_httpx_response(500), body=None), True),
        (lambda: APIStatusError("x", response=_httpx_response(502), body=None), True),
        (lambda: APIStatusError("x", response=_httpx_response(408), body=None), True),
        (lambda: APIStatusError("bad", response=_httpx_response(400), body=None), False),
        (lambda: APIStatusError("nf", response=_httpx_response(404), body=None), False),
        (lambda: RetryableHTTPStatusError(503, body_preview="x"), True),
        (lambda: RetryableHTTPStatusError(401, body_preview="no"), False),
        (
            lambda: httpx.HTTPStatusError(
                "err",
                request=httpx.Request("GET", "https://x"),
                response=_httpx_response(504),
            ),
            True,
        ),
        (
            lambda: httpx.HTTPStatusError(
                "err",
                request=httpx.Request("GET", "https://x"),
                response=_httpx_response(422),
            ),
            False,
        ),
        (
            lambda: httpx.ConnectError("down", request=httpx.Request("GET", "https://x")),
            True,
        ),
        (TimeoutError, True),
        (ConnectionError, True),
        (lambda: OSError(5, "io"), True),
        (lambda: ValueError("not transient"), False),
    ],
)
def test_is_retryable_error_predicate_matrix(
    factory: Callable[[], BaseException],
    expected: bool,
) -> None:
    exc = factory()
    assert is_retryable_error(exc) is expected
    assert is_openai_retryable(exc) is expected


def test_retryable_http_status_codes_is_expected_set() -> None:
    assert {408, 429, 500, 502, 503, 504} == RETRYABLE_HTTP_STATUS_CODES


# --- sleep_with_backoff ---


def test_sleep_with_backoff_exponential_and_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    def capture_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", capture_sleep)
    monkeypatch.setattr("cadence_md.app.retry_utils.random.random", lambda: 0.0)

    sleep_with_backoff(0, base_seconds=2.0, max_seconds=1000.0, jitter_ratio=0.1)
    sleep_with_backoff(3, base_seconds=2.0, max_seconds=10.0, jitter_ratio=0.0)

    assert sleeps == [2.0, 10.0]


def test_sleep_with_backoff_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", sleeps.append)
    monkeypatch.setattr("cadence_md.app.retry_utils.random.random", lambda: 1.0)

    sleep_with_backoff(1, base_seconds=10.0, max_seconds=100.0, jitter_ratio=0.2)

    # delay = 10 * 2^1 = 20, jitter = 20 * 0.2 * 1 = 4
    assert sleeps == [24.0]


# --- retry_sync (existing + edge cases) ---


def test_retry_sync_retries_retryable_http_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual HTTP wrappers can reuse the shared retry helper."""
    monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", MagicMock())
    attempts = {"count": 0}

    def op() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RetryableHTTPStatusError(500, body_preview="busy")
        return "ok"

    result = retry_sync(
        op,
        logger_=logging.getLogger(__name__),
        max_attempts=2,
        operation_name="test.http",
    )

    assert result == "ok"
    assert attempts["count"] == 2


def test_retry_sync_zero_attempts_means_single_call() -> None:
    """A misconfigured zero-attempt budget still performs the initial call."""
    op = MagicMock(return_value="ok")

    result = retry_sync(
        op,
        logger_=logging.getLogger(__name__),
        max_attempts=0,
        operation_name="test.once",
    )

    assert result == "ok"
    op.assert_called_once()


def test_retry_sync_does_not_retry_non_retryable_http_status() -> None:
    op = MagicMock(side_effect=RetryableHTTPStatusError(400, body_preview="bad request"))

    with pytest.raises(RetryableHTTPStatusError):
        retry_sync(
            op,
            logger_=logging.getLogger(__name__),
            max_attempts=3,
            operation_name="test.non_retryable",
        )

    op.assert_called_once()


def test_retry_sync_single_attempt_raises_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cadence_md.app.retry_utils.sleep_with_backoff", MagicMock())
    op = MagicMock(side_effect=RetryableHTTPStatusError(503))

    with pytest.raises(RetryableHTTPStatusError):
        retry_sync(
            op,
            logger_=logging.getLogger(__name__),
            max_attempts=1,
            operation_name="test.single",
        )

    op.assert_called_once()


def test_retry_sync_three_failures_then_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    backoff = MagicMock()
    monkeypatch.setattr("cadence_md.app.retry_utils.sleep_with_backoff", backoff)
    attempts = {"n": 0}

    def op() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RetryableHTTPStatusError(502)
        return "done"

    out = retry_sync(
        op,
        logger_=logging.getLogger(__name__),
        max_attempts=5,
        operation_name="test.waves",
    )

    assert out == "done"
    assert attempts["n"] == 3
    assert backoff.call_count == 2
    assert backoff.call_args_list[0].args[0] == 0
    assert backoff.call_args_list[1].args[0] == 1
    assert "retrying attempt 2/5" in caplog.text
    assert "retrying attempt 3/5" in caplog.text


def test_retry_sync_custom_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", MagicMock())
    attempts = {"n": 0}

    def op() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("transient for this test")
        return "ok"

    def only_value_error(exc: BaseException) -> bool:
        return isinstance(exc, ValueError)

    result = retry_sync(
        op,
        logger_=logging.getLogger(__name__),
        max_attempts=2,
        operation_name="test.custom",
        is_retryable=only_value_error,
    )

    assert result == "ok"
    assert attempts["n"] == 2


def test_retry_sync_custom_predicate_false_no_extra_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cadence_md.app.retry_utils.sleep_with_backoff", MagicMock())
    op = MagicMock(side_effect=RetryableHTTPStatusError(503))

    def never_retry(_: BaseException) -> bool:
        return False

    with pytest.raises(RetryableHTTPStatusError):
        retry_sync(
            op,
            logger_=logging.getLogger(__name__),
            max_attempts=5,
            operation_name="test.no_retry_predicate",
            is_retryable=never_retry,
        )

    op.assert_called_once()


def test_retry_sync_negative_max_attempts_normalized_to_one() -> None:
    op = MagicMock(return_value="x")

    retry_sync(
        op,
        logger_=logging.getLogger(__name__),
        max_attempts=-10,
        operation_name="test.norm",
    )

    op.assert_called_once()
