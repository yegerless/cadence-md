import asyncio
import logging
import random
import threading
import time
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import TypeVar

from gigachat.exceptions import ResponseError
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

from metrics.config import metrics_settings

T = TypeVar("T")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Interval between any GigaChat requests (seconds). If 429, reduce load or increase.
GIGACHAT_MIN_INTERVAL_SEC = metrics_settings.GIGACHAT_MIN_INTERVAL_SEC


def _is_gigachat_rate_limit(exc: BaseException) -> bool:
    """
    Check if the exception is a GigaChat rate limit error (429)

    Args:
        exc: Exception to check
    Returns:
        True if the exception is a GigaChat rate limit error
    """
    if isinstance(exc, ResponseError) and len(exc.args) >= 2:
        return int(exc.args[1]) == HTTPStatus.TOO_MANY_REQUESTS
    return False


class GigaChatApiThrottle:
    """
    Common queue for all GigaChat calls (chat + embeddings), otherwise 429 when interleaved
    """

    def __init__(self, min_interval_sec: float) -> None:
        """
        Initialize the GigaChat API throttle

        Args:
            min_interval_sec: Minimum interval between API calls to GigaChat
        """
        self._min_interval = min_interval_sec
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def wait(self) -> None:
        """
        Wait for the minimum interval between API calls to GigaChat
        """
        with self._lock:
            now = time.monotonic()
            wait_for = self._min_interval - (now - self._last_ts)
            if wait_for > 0:
                time.sleep(wait_for)
            self._last_ts = time.monotonic()


def _retry_sync[T](operation: Callable[[], T], *, logger_: logging.Logger) -> T:
    """
    Retry a synchronous operation if it fails with a GigaChat rate limit error

    Args:
        operation: Operation to retry
        logger_: Logger to use for logging
    Returns:
        Result of the operation
    """
    max_attempts = 12
    delay = 1.0
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except ResponseError as e:
            last = e
            if not _is_gigachat_rate_limit(e):
                raise
            logger_.warning(
                f"GigaChat 429 (sync), retry in {delay:.1f} seconds ({attempt}/{max_attempts})",
            )
            time.sleep(delay + random.uniform(0, 0.5 * delay))
            delay = min(60.0, delay * 2)
    assert last is not None
    raise last


async def _retry_async[T](
    factory: Callable[[], Awaitable[T]],
    *,
    logger_: logging.Logger,
) -> T:
    """
    Retry an asynchronous operation if it fails with a GigaChat rate limit error

    Args:
        factory: Factory to create the operation
        logger_: Logger to use for logging
    Returns:
        Result of the operation
    """
    max_attempts = 12
    delay = 1.0
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await factory()
        except ResponseError as e:
            last = e
            if not _is_gigachat_rate_limit(e):
                raise
            logger_.warning(
                f"GigaChat 429 (async), retry in {delay:.1f} seconds ({attempt}/{max_attempts})",
            )
            await asyncio.sleep(delay + random.uniform(0, 0.5 * delay))
            delay = min(60.0, delay * 2)
    assert last is not None
    raise last


GIGACHAT_API_THROTTLE = GigaChatApiThrottle(GIGACHAT_MIN_INTERVAL_SEC)


class ThrottledGigaChat(GigaChat):
    """
    Throttled GigaChat model (chat) to avoid 429 errors

    Args:
        messages: Messages to generate
        stop: Stop sequence for the generation
        run_manager: Run manager for the generation
        stream: Stream for the generation
        **kwargs: Additional arguments for the generation
    Returns:
        Result of the generation
    """

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        stream=None,
        **kwargs,
    ):
        """
        Generate a response from the GigaChat model

        Args:
            messages: Messages to generate
            stop: Stop sequence for the generation
            run_manager: Run manager for the generation
            stream: Stream for the generation
            **kwargs: Additional arguments for the generation
        Returns:
            Result of the generation
        """
        GIGACHAT_API_THROTTLE.wait()

        # Call the parent class to generate the response
        def _call():
            return super(ThrottledGigaChat, self)._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                stream=stream,
                **kwargs,
            )

        return _retry_sync(_call, logger_=logger)

    async def _agenerate(
        self,
        messages,
        stop=None,
        run_manager=None,
        stream=None,
        **kwargs,
    ):
        """
        Generate a response from the GigaChat model asynchronously

        Args:
            messages: Messages to generate
            stop: Stop sequence for the generation
            run_manager: Run manager for the generation
            stream: Stream for the generation
            **kwargs: Additional arguments for the generation
        Returns:
            Result of the generation
        """
        await asyncio.to_thread(GIGACHAT_API_THROTTLE.wait)

        async def _call():
            return await super(ThrottledGigaChat, self)._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                stream=stream,
                **kwargs,
            )

        return await _retry_async(_call, logger_=logger)


class ThrottledGigaChatEmbeddings(GigaChatEmbeddings):
    """
    Throttled GigaChat embeddings model to avoid 429 errors

    Args:
        texts: Texts to embed
    Returns:
        Embeddings for the texts
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts using the GigaChat embeddings model

        Args:
            texts: Texts to embed
        Returns:
            Embeddings for the texts
        """
        GIGACHAT_API_THROTTLE.wait()

        # Call the parent class to embed the texts
        def _call() -> list[list[float]]:
            return super(ThrottledGigaChatEmbeddings, self).embed_documents(texts)

        return _retry_sync(_call, logger_=logger)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts using the GigaChat embeddings model asynchronously

        Args:
            texts: Texts to embed
        Returns:
            Embeddings for the texts
        """
        await asyncio.to_thread(GIGACHAT_API_THROTTLE.wait)

        # Call the parent class to embed the texts asynchronously
        async def _call() -> list[list[float]]:
            return await super(ThrottledGigaChatEmbeddings, self).aembed_documents(texts)

        return await _retry_async(_call, logger_=logger)
