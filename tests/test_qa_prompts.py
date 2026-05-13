"""Integrity tests for QA generation prompt templates (no LLM)."""

from pathlib import Path

from qa_dataset_generator.config import SECTION_QUESTION_TYPES
from qa_dataset_generator.prompts import (
    _PROMPT_FILES,
    _PROMPTS_DIR,
    GENERATION_PROMPTS,
)


def _expected_generation_keys() -> set[tuple[str, str]]:
    """Keys implied by SECTION_QUESTION_TYPES (section value, question type value)."""
    keys: set[tuple[str, str]] = set()
    for section, question_types in SECTION_QUESTION_TYPES.items():
        for qt in question_types:
            keys.add((section.value, qt.value))
    return keys


def test_generation_prompts_match_section_question_config() -> None:
    expected = _expected_generation_keys()
    assert set(GENERATION_PROMPTS.keys()) == expected
    assert set(_PROMPT_FILES.keys()) == expected


def test_prompt_files_exist_on_disk() -> None:
    for file_name in _PROMPT_FILES.values():
        path = _PROMPTS_DIR / file_name
        assert path.is_file(), f"Missing prompt file: {path}"


def test_prompts_directory_has_no_extra_txt_files() -> None:
    on_disk = {p.name for p in _PROMPTS_DIR.glob("*.txt")}
    assert on_disk == set(_PROMPT_FILES.values())


def test_each_prompt_contains_context_placeholder_and_formats() -> None:
    probe = "__context_probe__"
    for key, template in GENERATION_PROMPTS.items():
        assert "{context}" in template, f"Missing {{context}} for {key}"
        formatted = template.format(context=probe)
        assert probe in formatted
        assert "{context}" not in formatted


def test_prompts_dir_is_next_to_prompts_module() -> None:
    prompts_module = Path(__file__).resolve().parents[1] / "qa_dataset_generator" / "prompts.py"
    assert prompts_module.parent / "prompts" == _PROMPTS_DIR
