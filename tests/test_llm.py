import sys
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from openai import RateLimitError

# Add parent directory to path for imports
sys.path.insert(0, "..")


import cadence_md.app.llm as llm_module
from cadence_md.app.llm import LLMWrapper, get_llm, get_llm_from_settings
from cadence_md.app.retry_utils import RetryableHTTPStatusError


class TestLLMWrapperInit:
    """Init tests for LLMWrapper"""

    def test_init_basic(self, mock_chat_model):
        """Test initialization with minimal parameters"""
        wrapper = LLMWrapper(chat_model=mock_chat_model)

        assert wrapper.chat_model == mock_chat_model

    def test_init_stores_chat_model(self, mock_chat_model):
        """Test that chat_model is stored"""
        wrapper = LLMWrapper(chat_model=mock_chat_model)

        assert wrapper.chat_model is mock_chat_model


class TestInvokeMessages:
    """``invoke_messages`` sends a message list with retries."""

    def test_invoke_messages_returns_content(self, mock_chat_model):
        fake_message = MagicMock()
        fake_message.content = "structured"
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        msgs = [SystemMessage(content="sys"), HumanMessage(content="user")]
        result = wrapper.invoke_messages(msgs)

        assert result == "structured"
        mock_chat_model.invoke.assert_called_once_with(msgs)


class TestInvoke:
    """Test methods for invoke()"""

    def test_invoke_returns_content(self, mock_chat_model):
        """Test that invoke() returns content"""
        # Настроим мок
        fake_message = MagicMock()
        fake_message.content = "test response"
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = wrapper.invoke("test prompt")

        assert result == "test response"
        mock_chat_model.invoke.assert_called_once()
        invoke_msgs = mock_chat_model.invoke.call_args[0][0]
        assert len(invoke_msgs) == 1
        assert isinstance(invoke_msgs[0], HumanMessage)
        assert invoke_msgs[0].content == "test prompt"

    def test_invoke_calls_chat_model(self, mock_chat_model):
        """Test that chat_model.invoke() is called with a single HumanMessage."""
        fake_message = MagicMock()
        fake_message.content = "test response"
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        wrapper.invoke("test prompt")

        mock_chat_model.invoke.assert_called_once()
        call_args = mock_chat_model.invoke.call_args[0]
        assert len(call_args[0]) == 1
        assert isinstance(call_args[0][0], HumanMessage)
        assert call_args[0][0].content == "test prompt"

    def test_invoke_handles_empty_content(self, mock_chat_model):
        """Test handling empty content"""
        fake_message = MagicMock()
        fake_message.content = ""
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = wrapper.invoke("test prompt")

        assert result == ""

    def test_invoke_calls_only_once_per_prompt(self, mock_chat_model):
        """Test single invoke call"""
        fake_message = MagicMock()
        fake_message.content = "test response"
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        wrapper.invoke("prompt 1")
        wrapper.invoke("prompt 2")

        assert mock_chat_model.invoke.call_count == 2


class TestStream:
    """Test methods for stream()"""

    def test_stream_yields_chunks(self, mock_chat_model):
        """Test streaming output from multiple chunks"""
        # Создаем мок для чанков
        chunk1 = MagicMock()
        chunk1.content = "Hello "
        chunk2 = MagicMock()
        chunk2.content = "world!"
        chunk3 = MagicMock()
        chunk3.content = None  # This should be skipped

        mock_chat_model.stream.return_value = [chunk1, chunk2, chunk3]

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = list(wrapper.stream("test prompt"))

        assert result == ["Hello ", "world!"]
        mock_chat_model.stream.assert_called_once()

    def test_stream_handles_none_content(self, mock_chat_model):
        """Test skipping chunks with None content"""
        chunk1 = MagicMock()
        chunk1.content = "First"
        chunk2 = MagicMock()
        chunk2.content = None
        chunk3 = MagicMock()
        chunk3.content = "Second"

        mock_chat_model.stream.return_value = [chunk1, chunk2, chunk3]

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = list(wrapper.stream("test prompt"))

        assert result == ["First", "Second"]

    def test_stream_retries_transient_error(self, mock_chat_model, monkeypatch: pytest.MonkeyPatch):
        """Transient stream failures are retried through the shared helper."""
        chunk = MagicMock()
        chunk.content = "Recovered"
        mock_chat_model.stream.side_effect = [TimeoutError("temporary"), [chunk]]
        monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", MagicMock())

        wrapper = LLMWrapper(chat_model=mock_chat_model, max_retries=2)
        result = list(wrapper.stream("test prompt"))

        assert result == ["Recovered"]
        assert mock_chat_model.stream.call_count == 2

    def test_stream_empty_response(self, mock_chat_model):
        """Test handling empty response"""
        mock_chat_model.stream.return_value = []

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = list(wrapper.stream("test prompt"))

        assert result == []

    def test_stream_sends_messages_to_chat_model(self, mock_chat_model):
        """Test sending messages to stream()"""

        chunk1 = MagicMock()
        chunk1.content = "test"

        mock_chat_model.stream.return_value = [chunk1]

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        list(wrapper.stream("test prompt"))

        # Test that stream was called with correct arguments
        mock_chat_model.stream.assert_called_once()
        call_args = mock_chat_model.stream.call_args[0]
        assert len(call_args[0]) == 1
        assert isinstance(call_args[0][0], HumanMessage)
        assert call_args[0][0].content == "test prompt"

    def test_stream_returns_iterable(self, mock_chat_model):
        """Test returning Iterable[str]"""
        chunk1 = MagicMock()
        chunk1.content = "test"

        mock_chat_model.stream.return_value = [chunk1]

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = wrapper.stream("test prompt")

        # Test that the result is an iterable
        assert hasattr(result, "__iter__")


class TestStreamMessages:
    """``stream_messages`` sends an explicit message list with retries."""

    def test_stream_messages_yields_chunks(self, mock_chat_model):
        chunk1 = MagicMock()
        chunk1.content = "Hello "
        chunk2 = MagicMock()
        chunk2.content = "world!"
        mock_chat_model.stream.return_value = [chunk1, chunk2]

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        msgs = [SystemMessage(content="sys"), HumanMessage(content="user")]
        result = list(wrapper.stream_messages(msgs))

        assert result == ["Hello ", "world!"]
        mock_chat_model.stream.assert_called_once_with(msgs)

    def test_stream_messages_retries_then_success(
        self, mock_chat_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", MagicMock())
        chunk = MagicMock()
        chunk.content = "done"
        mock_chat_model.stream.side_effect = [RetryableHTTPStatusError(502), [chunk]]

        wrapper = LLMWrapper(mock_chat_model, max_retries=3)
        msgs = [SystemMessage(content="s"), HumanMessage(content="u")]

        assert list(wrapper.stream_messages(msgs)) == ["done"]
        assert mock_chat_model.stream.call_count == 2
        assert mock_chat_model.stream.call_args[0][0] == msgs


class TestInvokeRetries:
    """``invoke`` uses :func:`retry_sync` for transient and rate-limit failures."""

    def test_invoke_retries_retryable_http_then_success(
        self, mock_chat_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", MagicMock())
        fake_ok = MagicMock()
        fake_ok.content = "recovered"
        mock_chat_model.invoke.side_effect = [RetryableHTTPStatusError(429), fake_ok]

        wrapper = LLMWrapper(mock_chat_model, max_retries=3)
        assert wrapper.invoke("prompt") == "recovered"
        assert mock_chat_model.invoke.call_count == 2

    def test_invoke_retries_openai_rate_limit_error(
        self, mock_chat_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", MagicMock())
        req = httpx.Request("POST", "http://inference/v1/chat/completions")
        resp = httpx.Response(429, request=req)
        rl = RateLimitError("too many requests", response=resp, body=None)
        fake_ok = MagicMock()
        fake_ok.content = "after limit"
        mock_chat_model.invoke.side_effect = [rl, fake_ok]

        wrapper = LLMWrapper(mock_chat_model, max_retries=3)
        assert wrapper.invoke("p") == "after limit"
        assert mock_chat_model.invoke.call_count == 2

    def test_invoke_non_retryable_raises_without_extra_attempts(
        self, mock_chat_model: MagicMock
    ) -> None:
        mock_chat_model.invoke.side_effect = ValueError("not transient")

        wrapper = LLMWrapper(mock_chat_model, max_retries=5)
        with pytest.raises(ValueError, match="not transient"):
            wrapper.invoke("p")

        assert mock_chat_model.invoke.call_count == 1

    def test_invoke_exhausts_retries(
        self, mock_chat_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", MagicMock())
        mock_chat_model.invoke.side_effect = RetryableHTTPStatusError(503)

        wrapper = LLMWrapper(mock_chat_model, max_retries=2)
        with pytest.raises(RetryableHTTPStatusError):
            wrapper.invoke("p")

        assert mock_chat_model.invoke.call_count == 2


class TestInvokeMessagesRetries:
    """``invoke_messages`` shares the same retry policy as ``invoke``."""

    def test_invoke_messages_retries_then_success(
        self, mock_chat_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", MagicMock())
        msgs = [SystemMessage(content="s"), HumanMessage(content="u")]
        fake_ok = MagicMock()
        fake_ok.content = "done"
        mock_chat_model.invoke.side_effect = [RetryableHTTPStatusError(502), fake_ok]

        wrapper = LLMWrapper(mock_chat_model, max_retries=3)
        assert wrapper.invoke_messages(msgs) == "done"
        assert mock_chat_model.invoke.call_count == 2
        assert mock_chat_model.invoke.call_args[0][0] == msgs


class TestStreamRetries:
    """Streaming buffers one full successful ``stream()`` pass before yielding."""

    def test_stream_retries_when_mid_iterate_raises(
        self, mock_chat_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("cadence_md.app.retry_utils.time.sleep", MagicMock())
        c1 = MagicMock()
        c1.content = "part-"
        c2 = MagicMock()
        c2.content = "full"

        def failing_first_pass() -> MagicMock:
            yield c1
            raise RetryableHTTPStatusError(503)

        mock_chat_model.stream.side_effect = [failing_first_pass(), [c1, c2]]

        wrapper = LLMWrapper(mock_chat_model, max_retries=2)
        assert list(wrapper.stream("q")) == ["part-", "full"]
        assert mock_chat_model.stream.call_count == 2


class TestGetLLM:
    """Test functions for get_llm()"""

    def test_get_llm_returns_llm_wrapper(self, mock_chat_model, settings):
        """Test the type of the returned object"""

        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                result = get_llm(
                    model=settings.rag_config.llm.model_name,
                    base_url=settings.MODEL_INFERENCE_BASE_URL,
                    api_key=settings.MODEL_INFERENCE_API_KEY,
                    temperature=settings.rag_config.llm.temperature,
                    max_completion_tokens=settings.rag_config.llm.max_new_tokens,
                    top_p=settings.rag_config.llm.top_p,
                    streaming=settings.rag_config.llm.streaming,
                )

                assert isinstance(result, LLMWrapper)
                assert result.chat_model == mock_chat_model
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_uses_settings(self, mock_chat_model, settings):
        """Test using settings from settings"""

        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                get_llm(
                    model=settings.rag_config.llm.model_name,
                    base_url=settings.MODEL_INFERENCE_BASE_URL,
                    api_key=settings.MODEL_INFERENCE_API_KEY,
                    temperature=settings.rag_config.llm.temperature,
                    max_completion_tokens=settings.rag_config.llm.max_new_tokens,
                    top_p=settings.rag_config.llm.top_p,
                    streaming=settings.rag_config.llm.streaming,
                    timeout_seconds=settings.rag_config.llm.timeout_seconds,
                )

                call_kwargs = llm_module.ChatOpenAI.call_args[1]

                assert call_kwargs["model"] == settings.rag_config.llm.model_name
                assert call_kwargs["base_url"] == settings.MODEL_INFERENCE_BASE_URL
                assert call_kwargs["api_key"] == settings.MODEL_INFERENCE_API_KEY
                assert call_kwargs["temperature"] == settings.rag_config.llm.temperature
                assert (
                    call_kwargs["max_completion_tokens"] == settings.rag_config.llm.max_new_tokens
                )
                assert call_kwargs["top_p"] == settings.rag_config.llm.top_p
                assert call_kwargs["streaming"] == settings.rag_config.llm.streaming
                assert call_kwargs["timeout"] == settings.rag_config.llm.timeout_seconds
                assert call_kwargs["max_retries"] == 0
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_streaming_true(self, mock_chat_model, settings):
        """Test that streaming=True is passed to ChatOpenAI"""

        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                get_llm(
                    model=settings.rag_config.llm.model_name,
                    base_url=settings.MODEL_INFERENCE_BASE_URL,
                    api_key=settings.MODEL_INFERENCE_API_KEY,
                    temperature=settings.rag_config.llm.temperature,
                    max_completion_tokens=settings.rag_config.llm.max_new_tokens,
                    top_p=settings.rag_config.llm.top_p,
                    streaming=True,
                )

                call_kwargs = llm_module.ChatOpenAI.call_args[1]
                assert call_kwargs["streaming"] is True
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_streaming_false_by_default(self, mock_chat_model, settings):
        """Test that streaming=False is passed when explicitly configured"""

        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                get_llm(
                    model=settings.rag_config.llm.model_name,
                    base_url=settings.MODEL_INFERENCE_BASE_URL,
                    api_key=settings.MODEL_INFERENCE_API_KEY,
                    temperature=settings.rag_config.llm.temperature,
                    max_completion_tokens=settings.rag_config.llm.max_new_tokens,
                    top_p=settings.rag_config.llm.top_p,
                    streaming=False,
                )

                call_kwargs = llm_module.ChatOpenAI.call_args[1]
                assert call_kwargs["streaming"] is False
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_default_streaming_false(self, mock_chat_model, settings):
        """Test that streaming=False from settings and creation of LLMWrapper"""

        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                result = get_llm(
                    model=settings.rag_config.llm.model_name,
                    base_url=settings.MODEL_INFERENCE_BASE_URL,
                    api_key=settings.MODEL_INFERENCE_API_KEY,
                    temperature=settings.rag_config.llm.temperature,
                    max_completion_tokens=settings.rag_config.llm.max_new_tokens,
                    top_p=settings.rag_config.llm.top_p,
                    streaming=settings.rag_config.llm.streaming,
                )

                call_kwargs = llm_module.ChatOpenAI.call_args[1]
                assert call_kwargs["streaming"] == settings.rag_config.llm.streaming

                assert isinstance(result, LLMWrapper)
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_from_settings(self, mock_chat_model, settings):
        """Settings builder passes LLM settings into the low-level factory."""
        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                result = get_llm_from_settings(settings)

                assert isinstance(result, LLMWrapper)
                call_kwargs = llm_module.ChatOpenAI.call_args[1]
                assert call_kwargs["model"] == settings.rag_config.llm.model_name
                assert call_kwargs["base_url"] == "http://test.local"
                assert call_kwargs["api_key"] == "test-key"
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_explicit_timeout_and_retry_backoff(
        self, mock_chat_model: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm_module, "ChatOpenAI", MagicMock(return_value=mock_chat_model))
        result = get_llm(
            model="custom",
            base_url="http://host/v1",
            api_key="key",
            temperature=0.2,
            max_completion_tokens=256,
            top_p=0.7,
            streaming=False,
            timeout_seconds=42.5,
            max_retries=7,
            backoff_base_seconds=2.5,
            backoff_max_seconds=90.0,
        )
        call_kwargs = llm_module.ChatOpenAI.call_args[1]
        assert call_kwargs["model"] == "custom"
        assert call_kwargs["base_url"] == "http://host/v1"
        assert call_kwargs["api_key"] == "key"
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_completion_tokens"] == 256
        assert call_kwargs["top_p"] == 0.7
        assert call_kwargs["streaming"] is False
        assert call_kwargs["timeout"] == 42.5
        assert call_kwargs["max_retries"] == 0
        assert result._max_retries == 7
        assert result._backoff_base_seconds == 2.5
        assert result._backoff_max_seconds == 90.0

    def test_get_llm_from_settings_maps_nested_llm_config(
        self, mock_chat_model: MagicMock, settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://inference/v1")
        monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "env-key")
        custom_llm = settings.rag_config.llm.model_copy(
            update={
                "model_name": "from-settings",
                "temperature": 0.11,
                "max_new_tokens": 2048,
                "top_p": 0.22,
                "streaming": True,
                "timeout_seconds": 33.0,
                "max_retries": 4,
                "backoff_base_seconds": 1.25,
                "backoff_max_seconds": 40.0,
            }
        )
        monkeypatch.setattr(
            settings,
            "rag_config",
            settings.rag_config.model_copy(update={"llm": custom_llm}),
        )
        monkeypatch.setattr(llm_module, "ChatOpenAI", MagicMock(return_value=mock_chat_model))

        wrapper = get_llm_from_settings(settings)
        ck = llm_module.ChatOpenAI.call_args[1]
        assert ck["model"] == "from-settings"
        assert ck["base_url"] == "http://inference/v1"
        assert ck["api_key"] == "env-key"
        assert ck["temperature"] == 0.11
        assert ck["max_completion_tokens"] == 2048
        assert ck["top_p"] == 0.22
        assert ck["streaming"] is True
        assert ck["timeout"] == 33.0
        assert isinstance(wrapper, LLMWrapper)
        assert wrapper._max_retries == 4
        assert wrapper._backoff_base_seconds == 1.25
        assert wrapper._backoff_max_seconds == 40.0
