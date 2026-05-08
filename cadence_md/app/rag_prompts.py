"""
Load and format versioned RAG prompt files next to this package.
"""

from __future__ import annotations

from pathlib import Path

from cadence_md.app.settings import settings

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_rag_system_prompt() -> str:
    """Load the raw system prompt text.

    May contain ``{prompt_version}`` for :func:`format_rag_system_prompt`.
    """
    return (_PROMPTS_DIR / "rag_system.txt").read_text(encoding="utf-8").strip()


def load_rag_user_prompt_template() -> str:
    """Return UTF-8 user prompt template.

    Placeholders: ``{context}``, ``{question}``.
    """
    return (_PROMPTS_DIR / "rag_user.txt").read_text(encoding="utf-8").strip()


def load_query_rewriter_system_prompt() -> str:
    """Load the system prompt for optional query rewriting."""
    return (_PROMPTS_DIR / "query_rewriter_system.txt").read_text(encoding="utf-8").strip()


def load_query_rewriter_user_prompt_template() -> str:
    """Return query rewriter user prompt template.

    Placeholders: ``{question}``, ``{retrieval_query}``, ``{rewrite_iteration}``,
    ``{context_relevance_score}``, ``{context_relevance_reason}``.
    """
    return (_PROMPTS_DIR / "query_rewriter_user.txt").read_text(encoding="utf-8").strip()


def load_context_relevance_system_prompt() -> str:
    """Load the system prompt for context relevance grading."""
    return (_PROMPTS_DIR / "context_relevance_system.txt").read_text(encoding="utf-8").strip()


def load_context_relevance_user_prompt_template() -> str:
    """Return context relevance grader user prompt template.

    Placeholders: ``{question}``, ``{retrieval_query}``, ``{context}``.
    """
    return (_PROMPTS_DIR / "context_relevance_user.txt").read_text(encoding="utf-8").strip()


def load_answer_formatter_system_prompt() -> str:
    """Load the system prompt for optional answer formatting."""
    return (_PROMPTS_DIR / "answer_formatter_system.txt").read_text(encoding="utf-8").strip()


def load_answer_formatter_user_prompt_template() -> str:
    """Return answer formatter user prompt template.

    Placeholders: ``{answer}``, ``{context}``, ``{sources}``.
    """
    return (_PROMPTS_DIR / "answer_formatter_user.txt").read_text(encoding="utf-8").strip()


def load_output_guardrails_system_prompt() -> str:
    """Load the system prompt for optional output guardrails."""
    return (_PROMPTS_DIR / "output_guardrails_system.txt").read_text(encoding="utf-8").strip()


def load_output_guardrails_user_prompt_template() -> str:
    """Return output guardrails user prompt template.

    Placeholders: ``{question}``, ``{retrieval_query}``, ``{context}``, ``{answer}``,
    ``{sources}``.
    """
    return (_PROMPTS_DIR / "output_guardrails_user.txt").read_text(encoding="utf-8").strip()


def format_rag_system_prompt() -> str:
    """Return the system prompt with ``prompt_version`` filled in.

    Used for telemetry fields and consistent UX around prompt iterations.
    """
    return load_rag_system_prompt().format(
        prompt_version=settings.rag_config.prompt_version,
    )
