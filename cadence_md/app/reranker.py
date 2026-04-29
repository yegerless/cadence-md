"""Document reranking via OpenAI-compatible ``POST /v1/rerank`` (llama.cpp reranker mode)."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RerankerAPIError(RuntimeError):
    """Raised when the rerank endpoint returns an unexpected or invalid payload."""


def _rerank_url(base_url: str) -> str:
    """
    Build the full ``/v1/rerank`` URL from an OpenAI-style base URL.

    Args:
        base_url: Typically ``http://host:port/v1`` (trailing slash allowed).

    Returns:
        Absolute URL for the rerank endpoint.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/rerank"
    return f"{base}/v1/rerank"


def _parse_rerank_response(
    data: Mapping[str, Any],
    *,
    expected_documents: int,
) -> list[float]:
    """
    Extract per-document scores in input order from a ``/v1/rerank`` JSON body.

    Args:
        data: Parsed JSON object from the rerank response.
        expected_documents: Number of input documents (length check).

    Returns:
        List of ``relevance_score`` values, one per input document index.

    Raises:
        RerankerAPIError: If ``results`` is missing, malformed, or incomplete.
    """
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        msg = f"Rerank response missing 'results' list: keys={list(data)!r}"
        raise RerankerAPIError(msg)

    scores: list[float | None] = [None] * expected_documents
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        try:
            idx = int(item["index"])
            score = float(item["relevance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"Invalid rerank result entry: {item!r}"
            raise RerankerAPIError(msg) from exc
        if idx < 0 or idx >= expected_documents:
            msg = f"Rerank result index out of range: index={idx}, n_docs={expected_documents}"
            raise RerankerAPIError(msg)
        scores[idx] = score

    if any(s is None for s in scores):
        missing = [i for i, s in enumerate(scores) if s is None]
        msg = f"Incomplete rerank results; missing indices: {missing}"
        raise RerankerAPIError(msg)

    return [float(s) for s in scores]


class RerankerWrapper:
    """
    Reranks LangChain documents using an OpenAI-compatible ``/v1/rerank`` endpoint.

    Designed for llama.cpp server with ``--reranking`` and a pooling=rank reranker model.
    Scores are the raw ``relevance_score`` values returned by the server (higher = more relevant).

    Args:
        model: Model id as expected by the inference server.
        top_k: Maximum number of documents to return after sorting by score.
        return_score: If False, scores in ``rerank`` are replaced with ``None``.
        base_url: OpenAI-compatible API base URL (e.g. ``http://127.0.0.1:1234/v1``).
        api_key: Bearer token for the server.
        timeout_s: HTTP timeout for a single rerank request.
        max_retries_on_rate_limit: Extra attempts on HTTP 429 with exponential backoff.
    """

    def __init__(
        self,
        model: str,
        top_k: int,
        return_score: bool,
        base_url: str,
        api_key: str,
        *,
        timeout_s: float = 120.0,
        max_retries_on_rate_limit: int = 8,
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.return_score = return_score
        self._api_url = _rerank_url(base_url)
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_retries_on_rate_limit = max_retries_on_rate_limit

    def _post_rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST JSON to the rerank endpoint with retries on rate limits.

        Args:
            payload: Request body (``model``, ``query``, ``documents``, optional ``top_n``).

        Returns:
            Parsed JSON object.

        Raises:
            RerankerAPIError: On repeated 429 exhaustion or non-success HTTP status.
            httpx.HTTPError: On transport errors after retries.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        for attempt in range(self._max_retries_on_rate_limit + 1):
            try:
                with httpx.Client(timeout=self._timeout_s) as client:
                    response = client.post(self._api_url, json=payload, headers=headers)
            except httpx.RequestError as exc:
                logger.warning("Rerank HTTP transport error: %s", exc)
                raise

            if response.status_code == 429:
                wait_s = min(120.0, 2.0**attempt)
                logger.warning(
                    "Rerank rate limited (429), retrying in %.1fs (attempt %s/%s)",
                    wait_s,
                    attempt + 1,
                    self._max_retries_on_rate_limit + 1,
                )
                if attempt >= self._max_retries_on_rate_limit:
                    msg = "Rerank endpoint returned 429 too many times"
                    raise RerankerAPIError(msg) from None
                time.sleep(wait_s)
                continue

            if response.status_code >= 400:
                body_preview = (response.text or "")[:500]
                msg = f"Rerank HTTP {response.status_code} for {self._api_url}: {body_preview!r}"
                raise RerankerAPIError(msg)

            try:
                return response.json()
            except ValueError as exc:
                msg = f"Rerank response is not valid JSON: {(response.text or '')[:200]!r}"
                raise RerankerAPIError(msg) from exc

        msg = "Rerank loop exited unexpectedly"
        raise RerankerAPIError(msg)

    def score_pairs(self, query: str, documents: list[Document]) -> list[float]:
        """
        Score each document against the query using ``/v1/rerank``.

        Args:
            query: User query string.
            documents: LangChain documents to score.

        Returns:
            One scalar score per document (same order as ``documents``).
        """
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": texts,
        }
        data = self._post_rerank(payload)
        return _parse_rerank_response(data, expected_documents=len(documents))

    def rerank(
        self, query: str, documents: list[Document]
    ) -> Sequence[tuple[Document, float | None]]:
        """
        Rerank documents by descending relevance score.

        Args:
            query: User query string.
            documents: Candidate documents.

        Returns:
            Documents sorted by score (descending), optionally with scores stripped.
        """
        if not documents:
            return []

        scores_list = self.score_pairs(query, documents)
        pairs = list(zip(documents, scores_list, strict=True))
        pairs.sort(key=lambda x: x[1], reverse=True)

        if self.top_k is not None:
            pairs = pairs[: self.top_k]

        if self.return_score:
            return pairs

        return [(doc, None) for doc, _ in pairs]


def get_reranker(
    model: str,
    top_k: int,
    return_score: bool,
    base_url: str,
    api_key: str,
    *,
    timeout_s: float = 120.0,
    max_retries_on_rate_limit: int = 8,
) -> RerankerWrapper:
    """
    Build a :class:`RerankerWrapper` for ``/v1/rerank``-based reranking.

    Args:
        model: Reranker model name on the OpenAI-compatible server.
        top_k: Max documents to return after reranking.
        return_score: Whether to expose scores in ``rerank`` results.
        base_url: Inference server base URL (``.../v1``).
        api_key: API key / bearer token.
        timeout_s: Per-request HTTP timeout in seconds.
        max_retries_on_rate_limit: Number of extra attempts after HTTP 429.

    Returns:
        Configured reranker instance.
    """
    return RerankerWrapper(
        model=model,
        top_k=top_k,
        return_score=return_score,
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
        max_retries_on_rate_limit=max_retries_on_rate_limit,
    )
