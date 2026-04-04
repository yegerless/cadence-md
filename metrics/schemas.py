from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class QATestCase:
    """Test case from a synthetic dataset"""

    question: str
    answer: str  # ground truth
    context: str  # reference context
    question_type: str
    section_type: str
    document_title: str
    mkb_codes: list[str]
    metadata: dict

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            question=data["question"],
            answer=data["answer"],
            context=data["context"],
            question_type=data["question_type"],
            section_type=data["section_type"],
            document_title=data.get("document_title", ""),
            mkb_codes=data.get("mkb_codes", []),
            metadata={
                "section_title": data.get("section_title", ""),
            },
        )


@dataclass
class RAGTestResult:
    """Test result for one question"""

    question: str
    ground_truth_answer: str
    ground_truth_context: str

    # Retriever results
    retrieved_contexts: list[Document]
    retrieval_scores: list[float]

    # Generator results
    generated_answer: str

    # Metadata
    question_type: str
    section_type: str
    test_case_id: int
