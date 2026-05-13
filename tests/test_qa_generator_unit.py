"""Unit tests for qa_dataset_generator.generator (no real LLM)."""

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest
from gigachat.exceptions import ResponseError

from qa_dataset_generator.config import MIN_ANSWER_LENGTH, MIN_QUESTION_LENGTH
from qa_dataset_generator.generator import (
    QAGenerationGraph,
    QAGenerationState,
    QAGeneratorNodes,
    _invoke_with_retry,
    _is_rate_limit_error,
    _is_retryable_error,
)
from qa_dataset_generator.schemas import QAPair
from tests.conftest import make_clinical_section


def test_is_rate_limit_error_true_for_429() -> None:
    err = ResponseError("too many", HTTPStatus.TOO_MANY_REQUESTS)
    assert _is_rate_limit_error(err) is True


def test_is_rate_limit_error_false_for_other_status() -> None:
    err = ResponseError("server err", HTTPStatus.INTERNAL_SERVER_ERROR)
    assert _is_rate_limit_error(err) is False


def test_is_rate_limit_error_false_for_non_response_error() -> None:
    assert _is_rate_limit_error(ValueError("x")) is False


def test_is_retryable_error_timeout_and_connection() -> None:
    assert _is_retryable_error(TimeoutError()) is True
    assert _is_retryable_error(ConnectionError()) is True


def test_is_retryable_error_value_error() -> None:
    assert _is_retryable_error(ValueError("bad")) is False


def test_invoke_with_retry_non_retryable_raises_immediately() -> None:
    chain = MagicMock()
    chain.invoke.side_effect = ValueError("fail")
    with pytest.raises(ValueError, match="fail"):
        _invoke_with_retry(chain, {})
    assert chain.invoke.call_count == 1


def test_invoke_with_retry_success_after_transient_failures() -> None:
    chain = MagicMock()
    chain.invoke.side_effect = [
        TimeoutError(),
        TimeoutError(),
        {"question": "q" * MIN_QUESTION_LENGTH, "answer": "a" * MIN_ANSWER_LENGTH},
    ]
    with patch("qa_dataset_generator.generator.time.sleep"):
        result = _invoke_with_retry(chain, {}, max_attempts=5)
    assert result["question"].startswith("q")
    assert chain.invoke.call_count == 3


def test_invoke_with_retry_unexpected_result_type_raises() -> None:
    chain = MagicMock()
    chain.invoke.return_value = "not a dict"
    with pytest.raises(ValueError, match="Unexpected result format"):
        _invoke_with_retry(chain, {})


def test_initialize_generation_sets_question_type_and_clears_errors() -> None:
    nodes = QAGeneratorNodes(MagicMock())
    section = make_clinical_section(section_type="diagnosis")
    state: QAGenerationState = {
        "section": section,
        "question_type": "",
        "generated_pair": MagicMock(),
        "errors": ["old"],
    }
    with patch("qa_dataset_generator.generator.random.choice", return_value="simple"):
        out = nodes.initialize_generation(state)
    assert out["question_type"] == "simple"
    assert out["generated_pair"] is None
    assert out["errors"] == []


def test_generate_questions_returns_early_when_question_type_empty() -> None:
    nodes = QAGeneratorNodes(MagicMock())
    section = make_clinical_section()
    state: QAGenerationState = {
        "section": section,
        "question_type": "",
        "generated_pair": None,
        "errors": [],
    }
    out = nodes.generate_questions(state)
    assert out["generated_pair"] is None
    assert out["errors"] == []


def test_generate_questions_missing_prompt_records_error() -> None:
    nodes = QAGeneratorNodes(MagicMock())
    section = make_clinical_section(section_type="treatment")
    state: QAGenerationState = {
        "section": section,
        "question_type": "nonexistent_qt",
        "generated_pair": None,
        "errors": [],
    }
    out = nodes.generate_questions(state)
    assert out["generated_pair"] is None
    assert any("Prompt not found" in e for e in out["errors"])


def test_generate_questions_success_builds_qa_pair() -> None:
    nodes = QAGeneratorNodes(MagicMock(), max_context_length=5000)
    section = make_clinical_section(section_type="treatment", section_title="Лечение")
    state: QAGenerationState = {
        "section": section,
        "question_type": "simple",
        "generated_pair": None,
        "errors": [],
    }
    payload = {
        "question": "Достаточно длинный вопрос для валидации",
        "answer": "Достаточно длинный ответ для прохождения проверки.",
        "context": "Цитата из контекста подтверждает ответ.",
    }
    with patch("qa_dataset_generator.generator._invoke_with_retry", return_value=payload):
        out = nodes.generate_questions(state)
    pair = out["generated_pair"]
    assert isinstance(pair, QAPair)
    assert pair.section_id == section.section_id
    assert pair.question_type == "simple"
    assert pair.section_type == "treatment"


def test_validate_qa_pair_missing_pair() -> None:
    nodes = QAGeneratorNodes(MagicMock())
    state: QAGenerationState = {
        "section": make_clinical_section(),
        "question_type": "simple",
        "generated_pair": None,
        "errors": [],
    }
    out = nodes.validate_qa_pair(state)
    assert out["generated_pair"] is None
    assert "generated_pair not found" in out["errors"][-1]


def test_validate_qa_pair_short_question() -> None:
    nodes = QAGeneratorNodes(MagicMock())
    pair = QAPair(
        question="short",
        answer="а" * MIN_ANSWER_LENGTH,
        question_type="simple",
        section_type="treatment",
        section_title="t",
        document_title="d",
        mkb_codes=[],
        context="long enough context for checks " + "x" * MIN_QUESTION_LENGTH,
        section_id="sid",
    )
    state: QAGenerationState = {
        "section": make_clinical_section(),
        "question_type": "simple",
        "generated_pair": pair,
        "errors": [],
    }
    out = nodes.validate_qa_pair(state)
    assert out["generated_pair"] is None
    assert any("Question too short" in e for e in out["errors"])


def test_validate_qa_pair_question_reused_from_context() -> None:
    nodes = QAGeneratorNodes(MagicMock())
    q_text = "Уникальный вопрос про терапию и длина достаточна"
    ctx = f"Введение. {q_text} Продолжение текста рекомендаций."
    pair = QAPair(
        question=q_text,
        answer="а" * MIN_ANSWER_LENGTH,
        question_type="simple",
        section_type="treatment",
        section_title="t",
        document_title="d",
        mkb_codes=[],
        context=ctx,
        section_id="sid",
    )
    state: QAGenerationState = {
        "section": make_clinical_section(),
        "question_type": "simple",
        "generated_pair": pair,
        "errors": [],
    }
    out = nodes.validate_qa_pair(state)
    assert out["generated_pair"] is None
    assert any("Question copy context" in e for e in out["errors"])


def test_validate_qa_pair_accepts_valid_pair() -> None:
    nodes = QAGeneratorNodes(MagicMock())
    pair = QAPair(
        question="Независимый клинический вопрос про тактику лечения?",
        answer="Ответ основан на разделе и имеет достаточную длину.",
        question_type="simple",
        section_type="treatment",
        section_title="t",
        document_title="d",
        mkb_codes=["B99"],
        context="Отдельная цитата из контекста без дублирования вопроса.",
        section_id="sid",
    )
    state: QAGenerationState = {
        "section": make_clinical_section(),
        "question_type": "simple",
        "generated_pair": pair,
        "errors": [],
    }
    out = nodes.validate_qa_pair(state)
    assert out["generated_pair"] == pair


def test_qa_generation_graph_run_end_to_end() -> None:
    graph = QAGenerationGraph(MagicMock())
    section = make_clinical_section(section_type="treatment")
    payload = {
        "question": "Достаточно длинный вопрос для валидации графа?",
        "answer": "Достаточно длинный ответ для прохождения проверки графа.",
        "context": "Цитата из контекста для проверки графа.",
    }
    with (
        patch("qa_dataset_generator.generator.random.choice", return_value="simple"),
        patch("qa_dataset_generator.generator._invoke_with_retry", return_value=payload),
    ):
        pair = graph.run(section)
    assert pair is not None
    assert pair.section_id == section.section_id
