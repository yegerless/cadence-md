"""
Cross-encoder style reranking over candidate chunks via ``POST /v1/rerank``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from langchain_core.documents import Document

from cadence_md.app.retry_utils import (
    RETRYABLE_HTTP_STATUS_CODES,
    RetryableHTTPStatusError,
    retry_sync,
)
from cadence_md.app.settings import Settings, settings

logger = logging.getLogger(__name__)


def load_reranker_query_instruction(instruction_path: Path) -> str:
    """Return UTF-8 task instruction from ``instruction_path`` for the rerank ``query`` field.

    Leading and trailing whitespace are stripped; line breaks and spacing inside the file are left
    as in the source. :meth:`RerankerWrapper.score_pairs` inserts one space between this text and
    the user query when ``use_query_instruction`` is True.
    """
    raw = instruction_path.read_text(encoding="utf-8")
    return raw.strip()


class RerankerAPIError(RuntimeError):
    """Rerank payload invalid or non-retryable HTTP; RAG may fall back to retrieval order."""


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
        max_retries_on_transport: Extra attempts on transport errors (timeouts, disconnects).
        backoff_base_seconds: Base delay for backoff (429 and transport).
        backoff_max_seconds: Maximum delay between retries.
        use_query_instruction: If True, prepend task text from ``query_instruction_path`` to the
            query string sent to ``/v1/rerank`` (only the ``query`` field; documents unchanged).
        query_instruction_path: UTF-8 file with the instruction; defaults from settings.
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
        max_retries_on_transport: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 120.0,
        use_query_instruction: bool = True,
        query_instruction_path: Path | None = None,
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.return_score = return_score
        self.use_query_instruction = use_query_instruction
        if query_instruction_path:
            self._query_instruction = load_reranker_query_instruction(query_instruction_path)
        else:
            self._query_instruction = None
        self._api_url = _rerank_url(base_url)
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_retries_on_rate_limit = max_retries_on_rate_limit
        self._max_retries_on_transport = max_retries_on_transport
        self._backoff_base_seconds = backoff_base_seconds
        self._backoff_max_seconds = backoff_max_seconds
        self._http_client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self._timeout_s)
        return self._http_client

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def __enter__(self) -> RerankerWrapper:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _post_rerank_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST JSON to the rerank endpoint once.

        Args:
            payload: Request body (``model``, ``query``, ``documents``, optional ``top_n``).

        Returns:
            Parsed JSON object.

        Raises:
            RetryableHTTPStatusError: On transient HTTP statuses.
            RerankerAPIError: On non-success HTTP status or invalid JSON.
            httpx.RequestError: On transport errors.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        response = self._get_client().post(self._api_url, json=payload, headers=headers)
        body_preview = (response.text or "")[:500]

        if response.status_code in RETRYABLE_HTTP_STATUS_CODES:
            raise RetryableHTTPStatusError(response.status_code, body_preview=body_preview)

        if response.status_code >= 400:
            msg = f"Rerank HTTP {response.status_code} for {self._api_url}: {body_preview!r}"
            raise RerankerAPIError(msg)

        try:
            return response.json()
        except ValueError as exc:
            msg = f"Rerank response is not valid JSON: {(response.text or '')[:200]!r}"
            raise RerankerAPIError(msg) from exc

    def _post_rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to the rerank endpoint with shared retry/backoff handling."""
        retry_counts = {"rate_limit": 0, "transport_or_http": 0}

        def consume_retry_budget(name: str, max_retries: int) -> bool:
            if retry_counts[name] >= max_retries:
                return False
            retry_counts[name] += 1
            return True

        def is_rerank_retryable(exc: BaseException) -> bool:
            if isinstance(exc, RetryableHTTPStatusError):
                if exc.status_code == 429:
                    return consume_retry_budget(
                        "rate_limit",
                        self._max_retries_on_rate_limit,
                    )

                return consume_retry_budget(
                    "transport_or_http",
                    self._max_retries_on_transport,
                )

            if isinstance(exc, httpx.RequestError):
                return consume_retry_budget(
                    "transport_or_http",
                    self._max_retries_on_transport,
                )

            return False

        try:
            return retry_sync(
                lambda: self._post_rerank_once(payload),
                logger_=logger,
                max_attempts=(1 + self._max_retries_on_rate_limit + self._max_retries_on_transport),
                operation_name="reranker.post",
                base_seconds=self._backoff_base_seconds,
                max_seconds=self._backoff_max_seconds,
                is_retryable=is_rerank_retryable,
            )
        except RetryableHTTPStatusError as exc:
            if exc.status_code == 429:
                msg = "Rerank endpoint returned 429 too many times"
            else:
                msg = f"Rerank endpoint returned HTTP {exc.status_code} too many times"
            raise RerankerAPIError(msg) from exc

    def score_pairs(
        self,
        query: str,
        documents: list[Document],
        *,
        document_texts: list[str] | None = None,
    ) -> list[float]:
        """
        Score each document against the query using ``/v1/rerank``.

        Args:
            query: User query string.
            documents: LangChain documents to score.
            document_texts: Optional per-document strings sent to the rerank API (same order).

        Returns:
            One scalar score per document (same order as ``documents``).
        """
        if not documents:
            return []

        if document_texts is not None and len(document_texts) != len(documents):
            msg = "document_texts length must match documents length"
            raise ValueError(msg)

        if document_texts is not None:
            texts = document_texts
        else:
            texts = [d.page_content for d in documents]
        query_for_api = query
        if self.use_query_instruction and self._query_instruction:
            query_for_api = f"{self._query_instruction} {query}"
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query_for_api,
            "documents": texts,
        }
        data = self._post_rerank(payload)
        return _parse_rerank_response(data, expected_documents=len(documents))

    def rerank(
        self,
        query: str,
        documents: list[Document],
        *,
        document_texts: list[str] | None = None,
    ) -> Sequence[tuple[Document, float | None]]:
        """
        Rerank documents by descending relevance score.

        Args:
            query: User query string.
            documents: Candidate documents.
            document_texts: Optional strings for reranking (e.g. titles + body); originals returned.

        Returns:
            Documents sorted by score (descending), optionally with scores stripped.
        """
        if not documents:
            return []

        scores_list = self.score_pairs(query, documents, document_texts=document_texts)
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
    max_retries_on_transport: int = 3,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 120.0,
    use_query_instruction: bool = True,
    query_instruction_path: Path | None = None,
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
        max_retries_on_transport: Extra attempts on transport errors.
        backoff_base_seconds: Backoff base delay.
        backoff_max_seconds: Backoff cap.
        use_query_instruction: Whether to prepend rerank task instruction to the query string.
        query_instruction_path: Path to the instruction file; defaults from ``RerankerConfig``.

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
        max_retries_on_transport=max_retries_on_transport,
        backoff_base_seconds=backoff_base_seconds,
        backoff_max_seconds=backoff_max_seconds,
        use_query_instruction=use_query_instruction,
        query_instruction_path=query_instruction_path,
    )


def get_reranker_from_settings(app_settings: Settings = settings) -> RerankerWrapper:
    """Build a reranker from ``rag_config.reranker`` and ``MODEL_INFERENCE_*`` env values."""
    cfg = app_settings.rag_config.reranker
    return get_reranker(
        model=cfg.model_name,
        top_k=cfg.top_k,
        return_score=cfg.return_score,
        base_url=app_settings.MODEL_INFERENCE_BASE_URL,
        api_key=app_settings.MODEL_INFERENCE_API_KEY,
        timeout_s=cfg.timeout_seconds,
        max_retries_on_rate_limit=cfg.max_retries_on_rate_limit,
        max_retries_on_transport=cfg.max_retries_on_transport,
        backoff_base_seconds=cfg.backoff_base_seconds,
        backoff_max_seconds=cfg.backoff_max_seconds,
        use_query_instruction=cfg.use_query_instruction,
        query_instruction_path=cfg.query_instruction_path,
    )
