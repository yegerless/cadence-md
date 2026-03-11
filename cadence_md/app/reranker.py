from openai import OpenAI

from cadence_md.app.settings import (
    MODEL_INFERENCE_API_KEY,
    MODEL_INFERENCE_BASE_URL,
    RAGConfig,
)


class BGERerankerWrapper:
    """
    A wrapper around bge-reranker-v2-m3, running via the OpenAI-compatible /embeddings endpoint.

    Semantics:
    - encode_pairs -> list of scores (float)
    - rerank -> sort candidates by score
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        return_score: bool = True,
    ):
        self.model = model
        self.return_score = return_score
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def _build_inputs(
        self,
        query: str,
        documents: list[str],
    ) -> list[str]:
        """
        BGE-reranker — cross-encoder: на каждую пару (q, d)
        отправляем конкатенированную строку или спец-формат.

        Здесь для универсальности используем простой join:
        '[query] [SEP] [doc]'. Если твой сервер ожидает другой формат,
        поменяй эту функцию.
        """
        return [f"{query} [SEP] {doc}" for doc in documents]

    def encode_pairs(
        self,
        queries: list[str],
        documents_list: list[list[str]],
    ) -> list[list[float]]:
        """
        Посчитать скоры для нескольких запросов и списков документов.

        queries: список запросов, длина N
        documents_list: список списков документов, длина N
        return: [[score_doc1, score_doc2, ...] для каждого query]
        """
        assert len(queries) == len(documents_list), (
            "queries и documents_list должны быть одинаковой длины"
        )

        all_inputs = []
        group_sizes = []

        for q, docs in zip(queries, documents_list, strict=True):
            inputs = self._build_inputs(q, docs)
            all_inputs.extend(inputs)
            group_sizes.append(len(docs))

        if not all_inputs:
            return [[] for _ in queries]

        # Вызываем OpenAI-совместимый /embeddings для bge-reranker-v2-m3
        # и интерпретируем embedding как скалярный скор (например, 1D-вектор).
        response = self.client.embeddings.create(
            model=self.model,
            input=all_inputs,
            encoding_format="float",
        )

        # Предполагаем, что сервер возвращает embedding как список из одного
        # числа или небольшой вектор, который можно скаляризовать.
        raw_vectors: list[list[float]] = [item.embedding for item in response.data]

        # Сводим к скаляру (например, берем первый элемент)
        scores_flat: list[float] = [
            float(vec[0]) if isinstance(vec, list) and len(vec) > 0 else float(vec)
            for vec in raw_vectors
        ]

        # Распаковываем обратно по запросам
        result: list[list[float]] = []
        offset = 0
        for size in group_sizes:
            result.append(scores_flat[offset : offset + size])
            offset += size

        return result

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Реранкает список documents для одного query.
        Возвращает список (document, score) отсортированный по убыванию score.
        """
        if not documents:
            return []

        scores_list = self.encode_pairs([query], [documents])[0]
        pairs = list(zip(documents, scores_list, strict=True))
        pairs.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            pairs = pairs[:top_k]

        if self.return_score:
            return pairs

        # Если score не нужен — вернём только тексты
        return [(doc, None) for doc, _ in pairs]


def get_reranker(config: RAGConfig) -> BGERerankerWrapper:
    """
    Фабрика по аналогии с get_embedder.
    """
    return BGERerankerWrapper(
        model=config.reranker.model_name,
        return_score=True,
        base_url=MODEL_INFERENCE_BASE_URL,
        api_key=MODEL_INFERENCE_API_KEY,
    )
