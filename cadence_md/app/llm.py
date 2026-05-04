"""
Chat completion wrapper around LangChain ``ChatOpenAI`` with uniform retries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from cadence_md.app.retry_utils import retry_sync
from cadence_md.app.settings import Settings, settings

logger = logging.getLogger(__name__)


class LLMWrapper:
    """
    Thin adapter over :class:`langchain_openai.ChatOpenAI` with synchronous retries.

    ``invoke`` accepts a raw string for simple call sites; ``invoke_messages`` is preferred for RAG
    (system + user messages). ``stream`` mirrors ``invoke`` for raw strings, while
    ``stream_messages`` streams an explicit message list.

    Args:
        chat_model: The ChatOpenAI model to wrap.
        max_retries: Total attempts (including the first) on transient errors.
        backoff_base_seconds: Base delay for exponential backoff between retries.
        backoff_max_seconds: Maximum delay between retries.
    """

    def __init__(
        self,
        chat_model: ChatOpenAI,
        *,
        max_retries: int = 6,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 120.0,
    ) -> None:
        self.chat_model = chat_model
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._backoff_max_seconds = backoff_max_seconds

    def invoke(self, prompt: str) -> str:
        """
        Run a single string prompt through the chat model (user-role content implied by the SDK).

        Args:
            prompt: Full user text for one-turn completion.

        Returns:
            Model response content as a string.
        """

        messages = [HumanMessage(content=prompt)]

        def call() -> str:
            resp = self.chat_model.invoke(messages)
            return str(resp.content)

        return retry_sync(
            call,
            logger_=logger,
            max_attempts=self._max_retries,
            operation_name="llm.invoke",
            base_seconds=self._backoff_base_seconds,
            max_seconds=self._backoff_max_seconds,
        )

    def invoke_messages(self, messages: list[BaseMessage]) -> str:
        """
        Invoke the chat model with an explicit message list (system + user, etc.).

        Args:
            messages: LangChain messages to send to the model.

        Returns:
            Model text content.
        """

        def call() -> str:
            resp = self.chat_model.invoke(messages)
            return str(resp.content)

        return retry_sync(
            call,
            logger_=logger,
            max_attempts=self._max_retries,
            operation_name="llm.invoke_messages",
            base_seconds=self._backoff_base_seconds,
            max_seconds=self._backoff_max_seconds,
        )

    def stream(self, prompt: str) -> Iterable[str]:
        """
        Stream a single string prompt (user-role content implied by ``HumanMessage``).

        Args:
            prompt: User message text.

        Yields:
            Successive string fragments from the model (implementation-defined chunking).
        """
        messages = [HumanMessage(content=prompt)]

        def call() -> list[str]:
            chunks: list[str] = []
            for chunk in self.chat_model.stream(messages):
                # Buffer a successful attempt so retries cannot duplicate partial output.
                if chunk.content:
                    chunks.append(str(chunk.content))
            return chunks

        yield from retry_sync(
            call,
            logger_=logger,
            max_attempts=self._max_retries,
            operation_name="llm.stream",
            base_seconds=self._backoff_base_seconds,
            max_seconds=self._backoff_max_seconds,
        )

    def stream_messages(self, messages: list[BaseMessage]) -> Iterable[str]:
        """
        Stream token/text chunks from an explicit message list with retry-safe buffering.

        The inner callable drains ``chat_model.stream`` into a list first; only after a full
        successful pass does the outer ``retry_sync`` return, and then this generator yields those
        chunks. That avoids emitting duplicate segments when the HTTP stream fails mid-way and the
        retry layer reruns the operation.

        Args:
            messages: LangChain messages to stream from the model.

        Yields:
            Successive string fragments from the model (implementation-defined chunking).
        """

        def call() -> list[str]:
            chunks: list[str] = []
            for chunk in self.chat_model.stream(messages):
                # Buffer a successful attempt so retries cannot duplicate partial output.
                if chunk.content:
                    chunks.append(str(chunk.content))
            return chunks

        yield from retry_sync(
            call,
            logger_=logger,
            max_attempts=self._max_retries,
            operation_name="llm.stream_messages",
            base_seconds=self._backoff_base_seconds,
            max_seconds=self._backoff_max_seconds,
        )


def get_llm(
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_completion_tokens: int,
    top_p: float,
    streaming: bool,
    *,
    timeout_seconds: float = 300.0,
    max_retries: int = 6,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 120.0,
) -> LLMWrapper:
    """
    Construct :class:`LLMWrapper` with explicit chat parameters.

    Args:
        model: The model name to use.
        base_url: The base URL of the model inference server.
        api_key: The API key to use.
        temperature: The temperature to use.
        max_completion_tokens: The maximum completion tokens to use.
        top_p: The top p to use.
        streaming: Whether to stream the output.
        timeout_seconds: Per-request timeout passed to the HTTP client.
        max_retries: Application-level retries on transient failures.
        backoff_base_seconds: Backoff base delay.
        backoff_max_seconds: Backoff cap.

    Returns:
        LLMWrapper: An instance of the LLMWrapper class.
    """
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        streaming=streaming,
        timeout=timeout_seconds,
        max_retries=0,
    )

    return LLMWrapper(
        llm,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        backoff_max_seconds=backoff_max_seconds,
    )


def get_llm_from_settings(app_settings: Settings = settings) -> LLMWrapper:
    """Build an LLM from ``Settings.rag_config.llm`` and ``MODEL_INFERENCE_*`` env configuration."""
    cfg = app_settings.rag_config.llm
    return get_llm(
        model=cfg.model_name,
        base_url=app_settings.MODEL_INFERENCE_BASE_URL,
        api_key=app_settings.MODEL_INFERENCE_API_KEY,
        temperature=cfg.temperature,
        max_completion_tokens=cfg.max_new_tokens,
        top_p=cfg.top_p,
        streaming=cfg.streaming,
        timeout_seconds=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
        backoff_base_seconds=cfg.backoff_base_seconds,
        backoff_max_seconds=cfg.backoff_max_seconds,
    )
