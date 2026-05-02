from langchain_core.documents import Document

from metrics.retrieval_utils import collect_missed_retrieval_case_ids
from metrics.schemas import RAGTestResult


def test_collect_missed_retrieval_case_ids_returns_missed_ids() -> None:
    results = [
        RAGTestResult(
            question="q1",
            ground_truth_answer="a1",
            ground_truth_context="aspirin daily 75mg for patient prevention",
            retrieved_contexts=[
                Document(page_content="irrelevant text", metadata={}),
                Document(page_content="completely different context", metadata={}),
            ],
            retrieval_scores=[0.4, 0.3],
            generated_answer="",
            question_type="factoid",
            section_type="therapy",
            test_case_id=1,
        ),
        RAGTestResult(
            question="q2",
            ground_truth_answer="a2",
            ground_truth_context="acetylsalicylic acid daily 75mg for patient prevention",
            retrieved_contexts=[
                Document(
                    page_content="acetylsalicylic acid daily 75mg for patient prevention",
                    metadata={},
                )
            ],
            retrieval_scores=[0.9],
            generated_answer="",
            question_type="factoid",
            section_type="therapy",
            test_case_id=2,
        ),
    ]

    missed = collect_missed_retrieval_case_ids(
        results=results,
        k=2,
        matcher=lambda gt, retrieved: gt in retrieved or retrieved in gt,
    )

    assert missed == [1]


def test_collect_missed_retrieval_case_ids_handles_empty_results() -> None:
    missed = collect_missed_retrieval_case_ids(
        results=[],
        k=5,
        matcher=lambda _gt, _retrieved: False,
    )
    assert missed == []
