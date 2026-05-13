"""Tests for GigaChat throttling and rate-limit retry helpers (no network)."""

import asyncio
import logging
import time
from http import HTTPStatus

import pytest
from gigachat.exceptions import ResponseError

from metrics.gigachat_api_wrapper import (
    GigaChatApiThrottle,
    _embedding_text_batches,
    _is_gigachat_rate_limit,
    _retry_async,
    _retry_sync,
    _truncate_embedding_text,
)


def test_is_gigachat_rate_limit_true_for_429() -> None:
    err = ResponseError("too many", HTTPStatus.TOO_MANY_REQUESTS)
    assert _is_gigachat_rate_limit(err) is True


def test_is_gigachat_rate_limit_false_for_other_status() -> None:
    err = ResponseError("server err", HTTPStatus.INTERNAL_SERVER_ERROR)
    assert _is_gigachat_rate_limit(err) is False


def test_is_gigachat_rate_limit_false_for_non_response_error() -> None:
    assert _is_gigachat_rate_limit(ValueError("x")) is False


def test_is_gigachat_rate_limit_false_for_short_response_error_args() -> None:
    err = ResponseError("only_one_arg")
    if len(err.args) < 2:
        assert _is_gigachat_rate_limit(err) is False


def test_gigachat_api_throttle_second_wait_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(float(s)))
    throttle = GigaChatApiThrottle(0.05)
    throttle.wait()
    throttle.wait()
    assert sleeps, "second wait() should sleep to respect min interval"
    assert sleeps[0] > 0


def test_truncate_embedding_text_keeps_payload_under_limit() -> None:
    text = "x" * 80

    trimmed = _truncate_embedding_text(text, max_chars=20)

    assert len(trimmed) <= 20
    assert trimmed.endswith("[...truncated]")


def test_embedding_text_batches_trim_and_split_by_total_chars() -> None:
    batches = _embedding_text_batches(
        ["a" * 12, "b" * 8, "ccc"],
        max_text_chars=5,
        max_batch_chars=10,
    )

    assert batches == [["a" * 5, "b" * 5], ["ccc"]]


def test_retry_sync_returns_without_retry() -> None:
    log = logging.getLogger("test_gigachat_sync")
    assert _retry_sync(lambda: 42, logger_=log) == 42


def test_retry_sync_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    log = logging.getLogger("test_gigachat_sync")
    calls = {"n": 0}

    def op() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ResponseError("rate", HTTPStatus.TOO_MANY_REQUESTS)
        return "ok"

    assert _retry_sync(op, logger_=log) == "ok"
    assert calls["n"] == 2


def test_retry_sync_non_429_response_error_raises_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    log = logging.getLogger("test_gigachat_sync")

    def op() -> None:
        raise ResponseError("bad", HTTPStatus.BAD_REQUEST)

    with pytest.raises(ResponseError, match="bad"):
        _retry_sync(op, logger_=log)


def test_retry_sync_exhausts_attempts_on_persistent_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
    log = logging.getLogger("test_gigachat_sync")
    calls = {"n": 0}

    def op() -> None:
        calls["n"] += 1
        raise ResponseError("rate", HTTPStatus.TOO_MANY_REQUESTS)

    with pytest.raises(ResponseError):
        _retry_sync(op, logger_=log)
    assert calls["n"] == 12


async def _noop_sleep(*_args: object, **_kwargs: object) -> None:
    return None


def test_retry_async_returns_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("metrics.gigachat_api_wrapper.asyncio.sleep", _noop_sleep)
    log = logging.getLogger("test_gigachat_async")

    async def factory() -> int:
        return 7

    assert asyncio.run(_retry_async(factory, logger_=log)) == 7


def test_retry_async_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("metrics.gigachat_api_wrapper.asyncio.sleep", _noop_sleep)
    log = logging.getLogger("test_gigachat_async")
    calls = {"n": 0}

    async def factory() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ResponseError("rate", HTTPStatus.TOO_MANY_REQUESTS)
        return "ok"

    assert asyncio.run(_retry_async(factory, logger_=log)) == "ok"
    assert calls["n"] == 2


def test_retry_async_non_429_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("metrics.gigachat_api_wrapper.asyncio.sleep", _noop_sleep)
    log = logging.getLogger("test_gigachat_async")

    async def factory() -> None:
        raise ResponseError("bad", HTTPStatus.BAD_REQUEST)

    with pytest.raises(ResponseError, match="bad"):
        asyncio.run(_retry_async(factory, logger_=log))


def test_retry_async_exhausts_on_persistent_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("metrics.gigachat_api_wrapper.asyncio.sleep", _noop_sleep)
    log = logging.getLogger("test_gigachat_async")
    calls = {"n": 0}

    async def factory() -> None:
        calls["n"] += 1
        raise ResponseError("rate", HTTPStatus.TOO_MANY_REQUESTS)

    with pytest.raises(ResponseError):
        asyncio.run(_retry_async(factory, logger_=log))
    assert calls["n"] == 12
