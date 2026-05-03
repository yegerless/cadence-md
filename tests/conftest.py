import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from cadence_md.app.pdf_parser.parser import ClinicalSection
from cadence_md.app.settings import settings as app_settings
from qa_dataset_generator.config import DEFAULT_MIN_CONTEXT_LENGTH

# Add parent directory to path for imports
sys.path.insert(0, "..")


@pytest.fixture
def settings():
    """Глобальный singleton настроек (тот же объект, что в cadence_md.app.settings)."""
    yield app_settings


@pytest.fixture
def mock_openai_client():
    """Mock для OpenAI клиента"""
    return MagicMock()


@pytest.fixture
def fake_response_data():
    """Фейковый response с OpenAI API"""
    return MagicMock(
        data=[
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
    )


@pytest.fixture
def fake_response_query():
    """Фейковый response для одного запроса"""
    return MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])


# --- LLM fixtures ---


@pytest.fixture
def mock_chat_model():
    """Mock для ChatOpenAI"""
    return MagicMock()


@pytest.fixture
def fake_ai_message():
    """Реальный AIMessage с content"""
    return AIMessage(content="test response")


@pytest.fixture
def fake_ai_message_chunk():
    """Реальный AIMessageChunk для stream()"""
    return AIMessageChunk(content="test response chunk")


# --- QA dataset generator ---

QA_LONG_CONTENT_MIN_LEN = DEFAULT_MIN_CONTEXT_LENGTH + 50


def make_clinical_section(
    *,
    filename: str = "guideline.pdf",
    document_title: str = "Клинические рекомендации",
    section_type: str = "treatment",
    section_title: str = "3. Лечение",
    content: str | None = None,
    mkb_codes: list[str] | None = None,
) -> ClinicalSection:
    """Секция с контентом длиннее порога отбора для QA-генерации."""
    if content is None:
        content = "а" * QA_LONG_CONTENT_MIN_LEN
    if mkb_codes is None:
        mkb_codes = ["A00"]
    return ClinicalSection(
        filename=filename,
        document_title=document_title,
        section_type=section_type,
        section_title=section_title,
        content=content,
        mkb_codes=mkb_codes,
    )


@pytest.fixture
def gigachat_dummy_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подставляет ключ API для конструктора QADatasetGenerator (без реального GigaChat)."""
    monkeypatch.setenv("GIGACHAT_API_KEY", "test-dummy-key")
    monkeypatch.setattr("qa_dataset_generator.generator.GIGACHAT_API_KEY", "test-dummy-key")


@pytest.fixture
def mock_gigachat_class(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Заменяет GigaChat на MagicMock в модуле генератора."""
    mock_cls = MagicMock()
    monkeypatch.setattr("qa_dataset_generator.generator.GigaChat", mock_cls)
    return mock_cls
