from pydantic import BaseModel, Field


class GeneratedQA(BaseModel):
    """Pydantic model for one QA pair (used for parsing the LLM response)"""

    question: str = Field(description="Вопрос врача")
    answer: str = Field(description="Ответ на основе клинических рекомендаций")
    context: str = Field(description="Цитата из текста, подтверждающая ответ")


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


class QAResponsePair(BaseModel):
    question: str = Field(description="Клинически релевантный вопрос по фрагменту рекомендаций")
    answer: str = Field(description="Короткий точный ответ, полностью основанный на фрагменте")
    context: str = Field(description="ТОЧНАЯ ЦИТАТА из фрагмента, подтверждающая ответ")
