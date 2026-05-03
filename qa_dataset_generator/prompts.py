"""A complete set of prompts for generating a QA dataset.

Prompt texts in ``qa_dataset_generator/prompts/*.txt``. This module owns the
explicit mapping from (section_type, question_type) keys to file names and
loads them eagerly at import time.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_PROMPT_FILES: dict[tuple[str, str], str] = {
    ("definition", "simple"): "definition_simple.txt",
    ("definition", "reasoning"): "definition_reasoning.txt",
    ("symptoms", "simple"): "symptoms_simple.txt",
    ("symptoms", "conditional"): "symptoms_conditional.txt",
    ("diagnosis", "simple"): "diagnosis_simple.txt",
    ("diagnosis", "comparison"): "diagnosis_comparison.txt",
    ("treatment", "simple"): "treatment_simple.txt",
    ("treatment", "conditional"): "treatment_conditional.txt",
    ("treatment", "comparison"): "treatment_comparison.txt",
    ("prevention", "simple"): "prevention_simple.txt",
    ("rehabilitation", "simple"): "rehabilitation_simple.txt",
}


def _load_prompt(file_name: str) -> str:
    """Read a UTF-8 prompt template from the prompts directory.

    Args:
        file_name: File name under ``qa_dataset_generator/prompts/``.

    Returns:
        Raw template string (may contain ``{context}`` and literal ``{{`` / ``}}``).
    """
    path = _PROMPTS_DIR / file_name
    return path.read_text(encoding="utf-8")


GENERATION_PROMPTS: dict[tuple[str, str], str] = {
    key: _load_prompt(file_name) for key, file_name in _PROMPT_FILES.items()
}
