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


def format_rag_system_prompt() -> str:
    """Return the system prompt with ``prompt_version`` filled in.

    Used for telemetry fields and consistent UX around prompt iterations.
    """
    return load_rag_system_prompt().format(
        prompt_version=settings.rag_config.prompt_version,
    )
