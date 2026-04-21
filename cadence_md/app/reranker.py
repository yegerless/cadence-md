from collections.abc import Iterable, Sequence

from langchain_core.documents import Document
from openai import OpenAI

from cadence_md.app.enums import RerankerAggregationStrategy


def _safe_vec(vec: Iterable[float]) -> list[float]:
    """
    Safe vector conversion.
    """
    v = list(vec)
    return v or [0.0]


def _l2_norm(vec: Iterable[float]) -> float:
    """
    L2 norm of a vector.
    """
    v = _safe_vec(vec)
    return float(sum(x * x for x in v) ** 0.5)


def _mean(vec: Iterable[float]) -> float:
    """
    Mean of a vector.
    """
    v = _safe_vec(vec)
    return float(sum(v) / len(v))


def _max_abs(vec: Iterable[float]) -> float:
    """
    Max absolute value of a vector.
    """
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
    Wrapper for the OpenAI reranker API.
    Reranks a list of documents based on a query.

    Args:
        model: The model name to use.
        instruction: Optional text prefixed to each embedding input (after
            stripping whitespace), before the ``query:`` / ``passage:`` block.
        top_k: The number of documents to return.
        return_score: Whether to return the score.
        embedding_agregation_strategy: The aggregation strategy to use.
        base_url: The base URL of the model inference server.
        api_key: The API key to use.
    """

    @staticmethod
    def _normalize_instruction(instruction: str | None) -> str | None:
        if instruction is None:
            return None
        stripped = instruction.strip()
        return stripped or None

    @staticmethod
    def _query_passage_body(query: str, passage: str) -> str:
        return f"query: {query}\npassage: {passage}"

    def __init__(
        self,
        model: str,
        instruction: str | None,
        top_k: int,
        return_score: bool,
        embedding_agregation_strategy: RerankerAggregationStrategy,
        base_url: str,
        api_key: str,
    ):
        self.model = model
        self.instruction = instruction
        self._instruction_for_prompt = self._normalize_instruction(instruction)
        self.top_k = top_k
        self.return_score = return_score
        self.embedding_agregation_strategy = embedding_agregation_strategy
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def _format_rerank_input(self, query: str, passage: str) -> str:
        body = self._query_passage_body(query, passage)
        prefix = self._instruction_for_prompt
        if prefix:
            return f"{prefix}\n{body}"
        return body

    def _build_inputs(
        self,
        query: str,
        documents: list[Document],
    ) -> list[str]:
        return [self._format_rerank_input(query, doc.page_content) for doc in documents]

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
        """ """
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


def get_reranker(
    model: str,
    instruction: str | None,
    top_k: int,
    return_score: bool,
    embedding_agregation_strategy: RerankerAggregationStrategy,
    base_url: str,
    api_key: str,
) -> RerankerWrapper:
    """
    Get a RerankerWrapper instance.

    Args:
        model: The model name to use.
        instruction: Optional per-input prefix for the embedding API.
        top_k: The number of documents to return.
        return_score: Whether to return the score.
        embedding_agregation_strategy: The aggregation strategy to use.
        base_url: The base URL of the model inference server.
        api_key: The API key to use.
    Returns:
        RerankerWrapper: An instance of the RerankerWrapper class.
    """

    return RerankerWrapper(
        model=model,
        instruction=instruction,
        top_k=top_k,
        return_score=return_score,
        embedding_agregation_strategy=embedding_agregation_strategy,
        base_url=base_url,
        api_key=api_key,
    )
