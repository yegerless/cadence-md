import numpy as np
from openai import OpenAI

from cadence_md.app.settings import settings


class EmbedderWrapper:
    """ """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        normalize: bool = True,
        return_score: bool = False,
    ):
        """ """
        self.model = model
        self.normalize = normalize
        self.return_score = return_score
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """ """
        response = self.client.embeddings.create(
            model=self.model, input=texts, encoding_format="float"
        )
        vectors = [item.embedding for item in response.data]

        if not self.normalize:
            return vectors

        arr = self._l2_normalize(np.array(vectors, dtype="float32"))
        return arr.tolist()

    def encode_query(self, query: str) -> list[float]:
        """ """
        response = self.client.embeddings.create(
            model=self.model, input=query, encoding_format="float"
        )
        query_vector = response.data[0].embedding

        if not self.normalize:
            return query_vector

        arr = self._l2_normalize(np.array(query_vector, dtype="float32"))
        return arr.tolist()

    def _l2_normalize(self, arr: np.ndarray) -> np.ndarray:
        """ """
        # for single 1-d vector
        if arr.ndim == 1:
            norm = np.linalg.norm(arr) + 1e-12
            return arr / norm
        # for matrix of 1-d vectors
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms


def get_embedder() -> EmbedderWrapper:
    """ """

    return EmbedderWrapper(
        model=settings.rag_config.embedding.model_name,
        normalize=settings.rag_config.embedding.normalize_embeddings,
        return_score=settings.rag_config.embedding.return_score,
        base_url=settings.MODEL_INFERENCE_BASE_URL,
        api_key=settings.MODEL_INFERENCE_API_KEY,
    )
