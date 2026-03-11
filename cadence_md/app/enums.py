from enum import StrEnum


class QuestionType(StrEnum):
    """Types of questions according to the RAGAS methodology"""

    SIMPLE = "simple"
    REASONING = "reasoning"
    MULTI_CONTEXT = "multi_context"
    CONDITIONAL = "conditional"
    COMPARISON = "comparison"


class SectionType(StrEnum):
    """Types of clinical recommendation sections"""

    DEFINITION = "definition"
    SYMPTOMS = "symptoms"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    PREVENTION = "prevention"
    REHABILITATION = "rehabilitation"
