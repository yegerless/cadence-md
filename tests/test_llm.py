import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

# Add parent directory to path for imports
sys.path.insert(0, "..")


import cadence_md.app.llm as llm_module
from cadence_md.app.llm import LLMWrapper, get_llm
from cadence_md.app.settings import settings


class TestLLMWrapperInit:
    """Тесты инициализации LLMWrapper"""

    def test_init_basic(self, mock_chat_model):
        """Проверяет инициализацию с минимальными параметрами"""
        wrapper = LLMWrapper(chat_model=mock_chat_model)

        assert wrapper.chat_model == mock_chat_model

    def test_init_stores_chat_model(self, mock_chat_model):
        """Проверяет, что chat_model сохраняется"""
        wrapper = LLMWrapper(chat_model=mock_chat_model)

        assert wrapper.chat_model is mock_chat_model


class TestInvoke:
    """Тесты метода invoke()"""

    def test_invoke_returns_content(self, mock_chat_model):
        """Проверяет, что invoke() возвращает content"""
        # Настроим мок
        fake_message = MagicMock()
        fake_message.content = "test response"
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = wrapper.invoke("test prompt")

        assert result == "test response"
        mock_chat_model.invoke.assert_called_once_with("test prompt")

    def test_invoke_calls_chat_model(self, mock_chat_model):
        """Проверяет вызов chat_model.invoke()"""
        fake_message = MagicMock()
        fake_message.content = "test response"
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        wrapper.invoke("test prompt")

        mock_chat_model.invoke.assert_called_once()
        call_args = mock_chat_model.invoke.call_args[0]
        assert call_args[0] == "test prompt"

    def test_invoke_handles_empty_content(self, mock_chat_model):
        """Проверяет обработку пустого content"""
        fake_message = MagicMock()
        fake_message.content = ""
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = wrapper.invoke("test prompt")

        assert result == ""

    def test_invoke_calls_only_once_per_prompt(self, mock_chat_model):
        """Проверяетsingle invoke call"""
        fake_message = MagicMock()
        fake_message.content = "test response"
        mock_chat_model.invoke.return_value = fake_message

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        wrapper.invoke("prompt 1")
        wrapper.invoke("prompt 2")

        assert mock_chat_model.invoke.call_count == 2


class TestStream:
    """Тесты метода stream()"""

    def test_stream_yields_chunks(self, mock_chat_model):
        """Проверяет потоковый вывод из нескольких чанков"""
        # Создаем мок для чанков
        chunk1 = MagicMock()
        chunk1.content = "Hello "
        chunk2 = MagicMock()
        chunk2.content = "world!"
        chunk3 = MagicMock()
        chunk3.content = None  # Этот должен быть пропущен

        mock_chat_model.stream.return_value = [chunk1, chunk2, chunk3]

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = list(wrapper.stream("test prompt"))

        assert result == ["Hello ", "world!"]
        mock_chat_model.stream.assert_called_once()

    def test_stream_handles_none_content(self, mock_chat_model):
        """Проверяет пропуск чанков с None content"""
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
        """Проверяет обработку пустого ответа"""
        mock_chat_model.stream.return_value = []

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = list(wrapper.stream("test prompt"))

        assert result == []

    def test_stream_sends_messages_to_chat_model(self, mock_chat_model):
        """Проверяет отправку сообщений в stream()"""

        chunk1 = MagicMock()
        chunk1.content = "test"

        mock_chat_model.stream.return_value = [chunk1]

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        list(wrapper.stream("test prompt"))

        # Проверяем, что stream был вызван с правильными аргументами
        mock_chat_model.stream.assert_called_once()
        call_args = mock_chat_model.stream.call_args[0]
        assert len(call_args[0]) == 1
        assert isinstance(call_args[0][0], HumanMessage)
        assert call_args[0][0].content == "test prompt"

    def test_stream_returns_iterable(self, mock_chat_model):
        """Проверяет возвращение Iterable[str]"""
        chunk1 = MagicMock()
        chunk1.content = "test"

        mock_chat_model.stream.return_value = [chunk1]

        wrapper = LLMWrapper(chat_model=mock_chat_model)
        result = wrapper.stream("test prompt")

        # Проверяем, что результат — это генератор
        assert hasattr(result, "__iter__")


class TestGetLLM:
    """Тесты функции get_llm()"""

    def test_get_llm_returns_llm_wrapper(self, mock_chat_model):
        """Проверяет тип возвращаемого объекта"""

        # Переопределим settings для теста
        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            mock_llm = MagicMock()
            mock_llm.chat_model = mock_chat_model

            # Переопределим ChatOpenAI

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                result = get_llm()

                assert isinstance(result, LLMWrapper)
                assert result.chat_model == mock_chat_model
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_uses_settings(self, mock_chat_model):
        """Проверяет использование настроек из settings"""

        with pytest.MonkeyPatch().context() as monkeypatch:
            # Настройка моков для settings
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            # Переопределение ChatOpenAI

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                get_llm()

                # Проверяем, что ChatOpenAI был вызван с правильными параметрами
                call_kwargs = llm_module.ChatOpenAI.call_args[1]

                assert call_kwargs["model"] == settings.rag_config.llm.model_name
                assert call_kwargs["base_url"] == settings.MODEL_INFERENCE_BASE_URL
                assert call_kwargs["api_key"] == settings.MODEL_INFERENCE_API_KEY
                assert call_kwargs["temperature"] == settings.rag_config.llm.temperature
                assert (
                    call_kwargs["max_completion_tokens"] == settings.rag_config.llm.max_new_tokens
                )
                assert call_kwargs["top_p"] == settings.rag_config.llm.top_p
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_streaming_true(self, mock_chat_model):
        """Проверяет streaming=True передаётся в ChatOpenAI"""

        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                get_llm(streaming=True)

                call_kwargs = llm_module.ChatOpenAI.call_args[1]
                assert call_kwargs["streaming"] is True
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_streaming_false_by_default(self, mock_chat_model):
        """Проверяет streaming=False по умолчанию"""

        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                get_llm()  # streaming=False по умолчанию

                call_kwargs = llm_module.ChatOpenAI.call_args[1]
                assert call_kwargs["streaming"] is False
            finally:
                llm_module.ChatOpenAI = original_chatopenai

    def test_get_llm_default_streaming_false(self, mock_chat_model):
        """Проверяет потStreamOpenAI вызывается streaming=False по умолчанию"""

        with pytest.MonkeyPatch().context() as monkeypatch:
            monkeypatch.setattr(settings, "MODEL_INFERENCE_BASE_URL", "http://test.local")
            monkeypatch.setattr(settings, "MODEL_INFERENCE_API_KEY", "test-key")

            original_chatopenai = llm_module.ChatOpenAI
            llm_module.ChatOpenAI = MagicMock(return_value=mock_chat_model)

            try:
                result = get_llm()  # No streaming argument — False by default

                call_kwargs = llm_module.ChatOpenAI.call_args[1]
                assert call_kwargs["streaming"] is False

                # Проверим, что LLMWrapper создан
                assert isinstance(result, LLMWrapper)
            finally:
                llm_module.ChatOpenAI = original_chatopenai
