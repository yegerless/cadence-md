import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

# Add parent directory to path for imports
sys.path.insert(0, "..")


import cadence_md.app.llm as llm_module
from cadence_md.app.llm import LLMWrapper, get_llm


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
        mock_chat_model.invoke.assert_called_once_with("test prompt")

    def test_invoke_calls_chat_model(self, mock_chat_model):
        """Test that chat_model.invoke() is called"""
        fake_message = MagicMock()
        fake_message.content = "test response"
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        wrapper.invoke("test prompt")

        mock_chat_model.invoke.assert_called_once()
        call_args = mock_chat_model.invoke.call_args[0]
        assert call_args[0] == "test prompt"

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
