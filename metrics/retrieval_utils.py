from collections.abc import Callable

from langchain_core.documents import Document

from metrics.schemas import RAGTestResult


def collect_missed_retrieval_case_ids(
    *,
    results: list[RAGTestResult],
    k: int,
    matcher: Callable[[RAGTestResult, Document], bool],
) -> list[int]:
    """
    Return IDs of cases where relevant context was missed in top-k

    Args:
        results: List of RAG test results
        k: Number of retrieved documents to evaluate
        matcher: Matcher to use for evaluation
    Returns:
        List of case IDs where relevant context was missed in top-k
    """
    missed_case_ids: list[int] = []
    for result in results:
        top_k_docs = result.retrieved_contexts[:k]
        has_match = any(matcher(result, doc) for doc in top_k_docs)
        if not has_match:
            missed_case_ids.append(result.test_case_id)
    return missed_case_ids
