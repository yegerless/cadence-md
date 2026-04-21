import numpy as np
from openai import OpenAI


class EmbedderWrapper:
    """
    Wrapper for the OpenAI embeddings API.
    Encodes text into a vector space.

    Args:
        model: The model name to use.
        api_key: The API key to use.
        base_url: The base URL of the model inference server.
        normalize: Whether to normalize the embeddings.
        return_score: Whether to return the score.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        normalize: bool = True,
        return_score: bool = False,
    ):
        """
        Initialize the EmbedderWrapper.

        Args:
            model: The model name to use.
            api_key: The API key to use.
            base_url: The base URL of the model inference server.
            normalize: Whether to normalize the embeddings.
            return_score: Whether to return the score.
        """
        self.model = model
        self.normalize = normalize
        self.return_score = return_score
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of text into a list of vectors.

        Args:
            texts: The list of text to encode.
        Returns:
            A list of vectors.
        """
        response = self.client.embeddings.create(
            model=self.model, input=texts, encoding_format="float"
        )
        vectors = [item.embedding for item in response.data]

        if not self.normalize:
            return vectors

        arr = self._l2_normalize(np.array(vectors, dtype="float32"))
        return arr.tolist()

    def encode_query(self, query: str) -> list[float]:
        """
        Encode a query into a vector.

        Args:
            query: The query to encode.
        Returns:
            A vector.
        """
        response = self.client.embeddings.create(
            model=self.model, input=query, encoding_format="float"
        )
        query_vector = response.data[0].embedding

        if not self.normalize:
            return query_vector

        arr = self._l2_normalize(np.array(query_vector, dtype="float32"))
        return arr.tolist()

    def _l2_normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        L2 normalize a vector.

        Args:
            arr: The vector to normalize.
        Returns:
            A normalized vector.
        """
        # for single 1-d vector
        if arr.ndim == 1:
            norm = np.linalg.norm(arr) + 1e-12
            return arr / norm
        # for matrix of 1-d vectors
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms


def get_embedder(
    model: str, normalize: bool, return_score: bool, base_url: str, api_key: str
) -> EmbedderWrapper:
    """
    Get an EmbedderWrapper instance.

    Args:
        model: The model name to use.
        normalize: Whether to normalize the embeddings.
        return_score: Whether to return the score.
        base_url: The base URL of the model inference server.
        api_key: The API key to use.
    Returns:
        EmbedderWrapper: An instance of the EmbedderWrapper class.
    """

    return EmbedderWrapper(
        model=model,
        normalize=normalize,
        return_score=return_score,
        base_url=base_url,
        api_key=api_key,
    )
