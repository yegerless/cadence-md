from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class GenerationPipelineStats:
    """Counters for the generation pipeline"""

    sections_loaded: int
    sections_selected: int
    skipped_invalid_jsonl: int
    sections_processed: int
    pairs_generated: int
    sections_failed: int
    min_context_length: int = 0
    sources_total: int = 0
    sources_skipped_short: int = 0
    skipped_sources: tuple[str, ...] = ()
    sections_filtered_short: int = 0


@dataclass
class _SelectionStats:
    """Internal counters for the section selection stage (before LLM invocation)."""

    sources_total: int = 0
    sources_skipped_short: int = 0
    skipped_sources: list[str] = field(default_factory=list)
    sections_filtered_short: int = 0


class QAPair(BaseModel):
    """The final structure of the QA pair for the dataset"""

    question: str
    answer: str
    question_type: str
    section_type: str
    section_title: str
    document_title: str
    mkb_codes: list[str]
    context: str
    section_id: str = ""


class QAResponsePair(BaseModel):
    """
    Схема для одной пары QA (используется для парсинга ответа LLM)

    Поля:
    - question: Клинически релевантный вопрос по фрагменту рекомендаций
    - answer: Короткий точный ответ, полностью основанный на фрагменте
    - context: ТОЧНАЯ ЦИТАТА из фрагмента, подтверждающая ответ

    Примечание: докстринг на русском языке, так как этот класс используется для парсинга ответа LLM
    """

    question: str = Field(description="Клинически релевантный вопрос по фрагменту рекомендаций")
    answer: str = Field(description="Короткий точный ответ, полностью основанный на фрагменте")
    context: str = Field(description="ТОЧНАЯ ЦИТАТА из фрагмента, подтверждающая ответ")
