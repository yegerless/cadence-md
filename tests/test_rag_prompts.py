"""Tests for :mod:`cadence_md.app.rag_prompts` prompt loading and formatting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cadence_md.app import rag_prompts


def test_load_rag_system_prompt_reads_utf8_and_strips(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "rag_system.txt").write_text("  \nСистема: {prompt_version}\n  \n", encoding="utf-8")
    monkeypatch.setattr(rag_prompts, "_PROMPTS_DIR", d)

    out = rag_prompts.load_rag_system_prompt()
    assert out == "Система: {prompt_version}"


def test_load_rag_user_prompt_template_reads_and_strips(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "prompts"
    d.mkdir()
    body = "Контекст:\n{context}\n\nВопрос: {question}"
    (d / "rag_user.txt").write_text(f"  \n{body}\n  ", encoding="utf-8")
    monkeypatch.setattr(rag_prompts, "_PROMPTS_DIR", d)

    assert rag_prompts.load_rag_user_prompt_template() == body


def test_load_output_guardrails_prompts_read_utf8_and_strip(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "prompts"
    d.mkdir()
    system_body = "Проверяй опору ответа на контекст"
    user_body = "Вопрос: {question}\nОтвет: {answer}\nИсточники: {sources}"
    (d / "output_guardrails_system.txt").write_text(
        f"  \n{system_body}\n  ",
        encoding="utf-8",
    )
    (d / "output_guardrails_user.txt").write_text(
        f"  \n{user_body}\n  ",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_prompts, "_PROMPTS_DIR", d)

    assert rag_prompts.load_output_guardrails_system_prompt() == system_body
    assert rag_prompts.load_output_guardrails_user_prompt_template() == user_body


def test_format_rag_system_prompt_fills_prompt_version(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "rag_system.txt").write_text("Версия промпта: {prompt_version}", encoding="utf-8")
    monkeypatch.setattr(rag_prompts, "_PROMPTS_DIR", d)

    fake_settings = SimpleNamespace(
        rag_config=SimpleNamespace(prompt_version="unit-test-2026"),
    )
    monkeypatch.setattr(rag_prompts, "settings", fake_settings)

    assert rag_prompts.format_rag_system_prompt() == "Версия промпта: unit-test-2026"


def test_prompt_loaders_use_expected_filenames(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure filenames stay ``rag_system.txt`` / ``rag_user.txt`` (contract for operators)."""
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "rag_system.txt").write_text("sys", encoding="utf-8")
    (d / "rag_user.txt").write_text("usr", encoding="utf-8")
    monkeypatch.setattr(rag_prompts, "_PROMPTS_DIR", d)

    assert rag_prompts.load_rag_system_prompt() == "sys"
    assert rag_prompts.load_rag_user_prompt_template() == "usr"


def test_optional_node_prompt_loaders_use_expected_filenames(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure optional RAG node prompt filenames stay stable for operators."""
    d = tmp_path / "prompts"
    d.mkdir()
    files = {
        "query_rewriter_system.txt": "qrs",
        "query_rewriter_user.txt": "question={question} retrieval={retrieval_query} "
        "iteration={rewrite_iteration} score={context_relevance_score} "
        "reason={context_relevance_reason}",
        "context_relevance_system.txt": "crs",
        "context_relevance_user.txt": "question={question} retrieval={retrieval_query} "
        "context={context}",
        "answer_formatter_system.txt": "afs",
        "answer_formatter_user.txt": "answer={answer} context={context} sources={sources}",
        "output_guardrails_system.txt": "ogs",
        "output_guardrails_user.txt": "question={question} retrieval={retrieval_query} "
        "context={context} answer={answer} sources={sources}",
    }
    for filename, text in files.items():
        (d / filename).write_text(text, encoding="utf-8")
    monkeypatch.setattr(rag_prompts, "_PROMPTS_DIR", d)

    assert rag_prompts.load_query_rewriter_system_prompt() == "qrs"
    assert "{retrieval_query}" in rag_prompts.load_query_rewriter_user_prompt_template()
    assert rag_prompts.load_context_relevance_system_prompt() == "crs"
    assert "{context}" in rag_prompts.load_context_relevance_user_prompt_template()
    assert rag_prompts.load_answer_formatter_system_prompt() == "afs"
    assert "{sources}" in rag_prompts.load_answer_formatter_user_prompt_template()
    assert rag_prompts.load_output_guardrails_system_prompt() == "ogs"
    assert "{answer}" in rag_prompts.load_output_guardrails_user_prompt_template()


def test_optional_node_prompt_files_have_expected_placeholders() -> None:
    assert "{question}" in rag_prompts.load_query_rewriter_user_prompt_template()
    assert "{retrieval_query}" in rag_prompts.load_query_rewriter_user_prompt_template()
    assert "{output_guardrail_score}" in rag_prompts.load_query_rewriter_user_prompt_template()
    assert "{output_guardrail_reason}" in rag_prompts.load_query_rewriter_user_prompt_template()
    assert (
        "{output_guardrail_unsupported_claims}"
        in rag_prompts.load_query_rewriter_user_prompt_template()
    )
    assert "{context}" in rag_prompts.load_context_relevance_user_prompt_template()
    assert "{answer}" in rag_prompts.load_answer_formatter_user_prompt_template()
    assert "{sources}" in rag_prompts.load_answer_formatter_user_prompt_template()
    assert "{question}" in rag_prompts.load_output_guardrails_user_prompt_template()
    assert "{retrieval_query}" in rag_prompts.load_output_guardrails_user_prompt_template()
    assert "{context}" in rag_prompts.load_output_guardrails_user_prompt_template()
    assert "{answer}" in rag_prompts.load_output_guardrails_user_prompt_template()
    assert "{sources}" in rag_prompts.load_output_guardrails_user_prompt_template()
