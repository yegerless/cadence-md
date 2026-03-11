from collections.abc import Iterable, Sequence

from langchain_core.documents import Document
from openai import OpenAI

from cadence_md.app.enums import RerankerAggregationStrategy
from cadence_md.app.settings import (
    MODEL_INFERENCE_API_KEY,
    MODEL_INFERENCE_BASE_URL,
    RAGConfig,
)


def _safe_vec(vec: Iterable[float]) -> list[float]:
    v = list(vec)
    return v if v else [0.0]


def _l2_norm(vec: Iterable[float]) -> float:
    v = _safe_vec(vec)
    return float(sum(x * x for x in v) ** 0.5)


def _mean(vec: Iterable[float]) -> float:
    v = _safe_vec(vec)
    return float(sum(v) / len(v))


def _max_abs(vec: Iterable[float]) -> float:
    v = _safe_vec(vec)
    return float(max((abs(x) for x in v), default=0.0))


def aggregate_embedding(
    vec: Iterable[float],
    strategy: RerankerAggregationStrategy,
) -> float:
    """
    Turns embedding into a scalar reranking score.
    """
    if strategy == RerankerAggregationStrategy.L2_NORM:
        return _l2_norm(vec)
    if strategy == RerankerAggregationStrategy.MEAN:
        return _mean(vec)
    if strategy == RerankerAggregationStrategy.MAX:
        return _max_abs(vec)

    # fallback — L2
    return _l2_norm(vec)


class RerankerWrapper:
    """
    A wrapper around reranker model (bge / qwen3-reranker / etc),
    running via the OpenAI-compatible /embeddings endpoint.

    Semantics:
    - encode_pairs -> list of scores (float)
    - rerank -> sort candidates by score
    """

    def __init__(
        self,
        model: str,
        instruction: str,
        api_key: str,
        base_url: str,
        embedding_agregation_strategy: RerankerAggregationStrategy,
        top_k: int | None = None,
        return_score: bool = True,
    ):
        self.model = model
        self.instruction = instruction
        self.top_k = top_k
        self.return_score = return_score
        self.embedding_agregation_strategy = embedding_agregation_strategy
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def _build_inputs(
        self,
        query: str,
        documents: list[Document],
    ) -> list[str]:
        return [(f"query: {query}\npassage: {doc.page_content}") for doc in documents]

    def encode_pairs(
        self,
        query: str,
        documents: list[Document],
    ) -> list[float]:
        prompts = self._build_inputs(query, documents)

        resp = self.client.embeddings.create(
            model=self.model,
            input=prompts,
            encoding_format="float",
        )

        scores: list[float] = []
        for item in resp.data:
            emb = item.embedding
            score = aggregate_embedding(
                emb,
                strategy=self.embedding_agregation_strategy,
            )
            scores.append(score)

        return scores

    def rerank(
        self, query: str, documents: list[Document]
    ) -> Sequence[tuple[Document, float | None]]:
        """
        Reranks a list of documents for a single query.
        Returns a list of (document, score) sorted by score in descending order.
        """
        if not documents:
            return []

        scores_list = self.encode_pairs(query, documents)
        pairs = list(zip(documents, scores_list, strict=True))
        pairs.sort(key=lambda x: x[1], reverse=True)

        if self.top_k is not None:
            pairs = pairs[: self.top_k]

        if self.return_score:
            return pairs

        return [(doc, None) for doc, _ in pairs]


def get_reranker(config: RAGConfig) -> RerankerWrapper:
    return RerankerWrapper(
        model=config.reranker.model_name,
        instruction=config.reranker.instruction,
        top_k=config.reranker.top_k,
        return_score=config.reranker.return_score,
        embedding_agregation_strategy=config.reranker.embedding_agregation_strategy,
        base_url=MODEL_INFERENCE_BASE_URL,
        api_key=MODEL_INFERENCE_API_KEY,
    )
