"""String enums for QA dataset generation (RAGAS-style question categories)."""

from enum import StrEnum


class QuestionType(StrEnum):
    """RAGAS-style question categories for QA dataset generation and metrics."""

    SIMPLE = "simple"
    REASONING = "reasoning"
    MULTI_CONTEXT = "multi_context"
    CONDITIONAL = "conditional"
    COMPARISON = "comparison"
