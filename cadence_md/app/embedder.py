"""Dense embeddings via an OpenAI-compatible HTTP API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from cadence_md.app.retry_utils import retry_sync
from cadence_md.app.settings import Settings, settings

logger = logging.getLogger(__name__)


def load_embedding_query_instruction(instruction_path: Path) -> str:
    """Load the optional query-side instruction text used only in ``encode_query``.

    The file is plain UTF-8 (often one line). Leading and trailing whitespace are stripped so
    editors and pre-commit hooks do not accidentally leave extra newlines; internal spacing is kept.
    :meth:`EmbedderWrapper.encode_query` concatenates this prefix, one space, and the user query.

    Args:
        instruction_path: Path to the UTF-8 instruction file; mirrors ``EmbeddingConfig`` defaults.

    Returns:
        Stripped instruction string, possibly empty if the file is blank after strip.
    """
    raw = instruction_path.read_text(encoding="utf-8")
    return raw.strip()


class EmbedderWrapper:
    """HTTP client for dense vectors used by Qdrant and shared with the rest of the stack.

    The OpenAI Python SDK uses ``max_retries=0``; :func:`~cadence_md.app.retry_utils.retry_sync`
    centralizes backoff and logging together with the LLM and reranker. L2 normalization keeps
    cosine similarity meaningful if the server returns raw vectors. Query-side instruction text
    (when enabled) applies only in :meth:`encode_query`, not in :meth:`encode`, so indexed chunks
    stay aligned with typical bi-encoder usage.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        normalize: bool = True,
        return_score: bool = False,
        *,
        use_query_instruction: bool = True,
        query_instruction_path: Path | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 6,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 120.0,
    ) -> None:
        """
        Initialize the EmbedderWrapper.

        Args:
            model: The model name to use.
            api_key: The API key to use.
            base_url: The base URL of the model inference server.
            normalize: Whether to normalize the embeddings.
            return_score: Whether to return the score.
            use_query_instruction: If True, prepend retrieval instruction text to queries in
                ``encode_query`` only (corpus indexing via ``encode`` is unchanged).
            query_instruction_path: UTF-8 text file with the query prefix; defaults from settings.
            timeout_seconds: HTTP timeout for embedding requests.
            max_retries: Total attempts (including the first) on transient errors.
            backoff_base_seconds: Base delay for exponential backoff between retries.
            backoff_max_seconds: Maximum delay between retries.
        """
        self.model = model
        self.normalize = normalize
        self.return_score = return_score
        self.use_query_instruction = use_query_instruction
        if query_instruction_path:
            self._query_instruction = load_embedding_query_instruction(query_instruction_path)
        else:
            self._query_instruction = None
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._backoff_max_seconds = backoff_max_seconds
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Embed many chunk texts in one API call.

        Used for indexing PDF chunks. Deliberately **does not** apply the query instruction prefix,
        so stored vectors align with training-style "document" representations.

        Args:
            texts: Chunk strings in batch order.

        Returns:
            One embedding per input row; same order as ``texts``.
        """

        def call() -> Any:
            return self.client.embeddings.create(
                model=self.model, input=texts, encoding_format="float"
            )

        response = retry_sync(
            call,
            logger_=logger,
            max_attempts=self._max_retries,
            operation_name="embeddings.encode",
            base_seconds=self._backoff_base_seconds,
            max_seconds=self._backoff_max_seconds,
        )
        vectors = [item.embedding for item in response.data]

        if not self.normalize:
            return vectors

        arr = self._l2_normalize(np.array(vectors, dtype="float32"))
        return arr.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Embed a single user query for dense retrieval.

        When enabled, prepends the loaded instruction and a single space before the user text, then
        calls the embeddings API once. This mirrors instruction-tuned retrieval behavior without
        changing indexed chunk vectors.

        Args:
            query: End-user question string.

        Returns:
            One dense vector (list of floats).
        """
        if self.use_query_instruction and self._query_instruction:
            text_for_api = f"{self._query_instruction} {query}"
        else:
            text_for_api = query

        def call() -> Any:
            return self.client.embeddings.create(
                model=self.model, input=text_for_api, encoding_format="float"
            )

        response = retry_sync(
            call,
            logger_=logger,
            max_attempts=self._max_retries,
            operation_name="embeddings.encode_query",
            base_seconds=self._backoff_base_seconds,
            max_seconds=self._backoff_max_seconds,
        )
        query_vector = response.data[0].embedding

        if not self.normalize:
            return query_vector

        arr = self._l2_normalize(np.array(query_vector, dtype="float32"))
        return arr.tolist()

    def _l2_normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        L2-normalize rows or a single vector; epsilon avoids division by zero.

        Args:
            arr: 1-D vector or 2-D matrix (one row per embedding).

        Returns:
            Same shape as ``arr`` with unit L2 norm per row (or whole vector if 1-D).
        """
        # for single 1-d vector
        if arr.ndim == 1:
            norm = np.linalg.norm(arr) + 1e-12
            return arr / norm
        # for matrix of 1-d vectors
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms


def get_embedder(
    model: str,
    normalize: bool,
    return_score: bool,
    base_url: str,
    api_key: str,
    *,
    use_query_instruction: bool = True,
    query_instruction_path: Path | None = None,
    timeout_seconds: float = 120.0,
    max_retries: int = 6,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 120.0,
) -> EmbedderWrapper:
    """
    Factory for :class:`EmbedderWrapper` with explicit parameters (tests and scripts).

    Args:
        model: The model name to use.
        normalize: Whether to normalize the embeddings.
        return_score: Whether to return the score.
        base_url: The base URL of the model inference server.
        api_key: The API key to use.
        use_query_instruction: Whether to prepend the retrieval instruction for ``encode_query``.
        query_instruction_path: Path to the query-prefix file; defaults from ``EmbeddingConfig``.
        timeout_seconds: Per-request timeout.
        max_retries: Retry attempts on transient failures.
        backoff_base_seconds: Backoff base delay.
        backoff_max_seconds: Backoff cap.

    Returns:
        EmbedderWrapper: An instance of the EmbedderWrapper class.
    """

    return EmbedderWrapper(
        model=model,
        normalize=normalize,
        return_score=return_score,
        base_url=base_url,
        api_key=api_key,
        use_query_instruction=use_query_instruction,
        query_instruction_path=query_instruction_path,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        backoff_max_seconds=backoff_max_seconds,
    )


def get_embedder_from_settings(app_settings: Settings = settings) -> EmbedderWrapper:
    """Construct an embedder from ``Settings.rag_config.embedding`` and inference env URLs.

    Keeps a single source of truth for model name, timeouts, retries, and query instruction path so
    CLI, metrics, and future services do not duplicate field wiring.
    """
    cfg = app_settings.rag_config.embedding
    return get_embedder(
        model=cfg.model_name,
        normalize=cfg.normalize_embeddings,
        return_score=cfg.return_score,
        base_url=app_settings.MODEL_INFERENCE_BASE_URL,
        api_key=app_settings.MODEL_INFERENCE_API_KEY,
        use_query_instruction=cfg.use_query_instruction,
        query_instruction_path=cfg.query_instruction_path,
        timeout_seconds=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
        backoff_base_seconds=cfg.backoff_base_seconds,
        backoff_max_seconds=cfg.backoff_max_seconds,
    )
