from dataclasses import dataclass, field

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
    section_id: str = ""

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a QATestCase from a dictionary

        Args:
            data: Dictionary containing the test case data
        Returns:
            QATestCase object
        """
        metadata = {
            "section_title": data.get("section_title", ""),
        }
        section_id = data.get("section_id", "")
        if section_id:
            metadata["section_id"] = section_id
        return cls(
            question=data["question"],
            answer=data["answer"],
            context=data["context"],
            question_type=data["question_type"],
            section_type=data["section_type"],
            document_title=data.get("document_title", ""),
            mkb_codes=data.get("mkb_codes", []),
            metadata=metadata,
            section_id=section_id,
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
    ground_truth_section_id: str = ""
    retrieval_query: str | None = None
    rewritten_queries: list[str] = field(default_factory=list)
    rag_flags: dict[str, bool] = field(default_factory=dict)
    context_relevance_score: float | None = None
