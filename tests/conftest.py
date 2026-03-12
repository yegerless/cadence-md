import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

# Add parent directory to path for imports
sys.path.insert(0, "..")


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
