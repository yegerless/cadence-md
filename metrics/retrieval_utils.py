from collections.abc import Callable

from metrics.schemas import RAGTestResult


def collect_missed_retrieval_case_ids(
    *,
    results: list[RAGTestResult],
    k: int,
    matcher: Callable[[str, str], bool],
) -> list[int]:
    """Return IDs of cases where relevant context was missed in top-k."""
    missed_case_ids: list[int] = []
    for result in results:
        gt_norm = result.ground_truth_context.lower().strip()
        top_k_docs = result.retrieved_contexts[:k]
        has_match = any(matcher(gt_norm, doc.page_content.lower().strip()) for doc in top_k_docs)
        if not has_match:
            missed_case_ids.append(result.test_case_id)
    return missed_case_ids
