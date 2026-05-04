import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from openai import RateLimitError

# Add parent directory to path for imports
sys.path.insert(0, "..")

from cadence_md.app.embedder import (
    EmbedderWrapper,
    get_embedder,
    get_embedder_from_settings,
    load_embedding_query_instruction,
)

_DEFAULT_QUERY_INSTRUCTION_PATH = (
    Path(__file__).resolve().parent.parent
    / "cadence_md"
    / "app"
    / "prompts"
    / "bge_m3_embedding_query_instruction.txt"
)


class TestLoadEmbeddingQueryInstruction:
    """Тесты load_embedding_query_instruction."""

    def test_strips_utf8_whitespace(self, tmp_path: Path) -> None:
        path = tmp_path / "instr.txt"
        path.write_text("  prefix line one.\n", encoding="utf-8")
        assert load_embedding_query_instruction(path) == "prefix line one."

    def test_blank_file_returns_empty_string(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.txt"
        path.write_text("  \n\t  ", encoding="utf-8")
        assert load_embedding_query_instruction(path) == ""


class TestEmbedderWrapperInit:
    """Тесты инициализации EmbedderWrapper"""

    def test_init_basic(self, mock_openai_client):
        """Проверяет инициализацию с минимальными параметрами"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )

        assert embedder.model == "test-model"
        assert embedder.normalize is True
        assert embedder.return_score is False

    def test_init_with_custom_params(self, mock_openai_client):
        """Проверяет инициализацию с кастомными параметрами"""
        embedder = EmbedderWrapper(
            model="custom-model",
            api_key="custom-key",
            base_url="http://custom-url.com",
            normalize=False,
            return_score=True,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )

        assert embedder.model == "custom-model"
        assert embedder.normalize is False
        assert embedder.return_score is True

    def test_init_default_normalize_true(self, mock_openai_client):
        """Проверяет, что normalize по умолчанию True"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        assert embedder.normalize is True

    def test_init_default_return_score_false(self, mock_openai_client):
        """Проверяет, что return_score по умолчанию False"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        assert embedder.return_score is False


class TestEncode:
    """Тесты метода encode"""

    def test_encode_with_normalize(self, mock_openai_client, fake_response_data):
        """Проверяет кодирование с нормализацией"""
        mock_openai_client.embeddings.create.return_value = fake_response_data

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        texts = ["test text 1", "test text 2"]
        result = embedder.encode(texts)

        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3

        # Проверяем нормализацию (сумма квадратов ≈ 1)
        for vector in result:
            norm = np.linalg.norm(vector)
            assert pytest.approx(norm, abs=1e-6) == 1.0

    def test_encode_without_normalize(self, mock_openai_client, fake_response_data):
        """Проверяет кодирование без нормализации"""
        mock_openai_client.embeddings.create.return_value = fake_response_data

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            normalize=False,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        texts = ["test text 1", "test text 2"]
        result = embedder.encode(texts)

        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3

        # Проверяем, что нормализация не применялась
        assert pytest.approx(np.linalg.norm(result[0]), abs=1e-06) != 1.0

    def test_encode_calls_api_with_correct_params(self, mock_openai_client, fake_response_data):
        """Проверяет вызов API с правильными параметрами"""
        mock_openai_client.embeddings.create.return_value = fake_response_data

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        texts = ["test text 1", "test text 2"]
        embedder.encode(texts)

        mock_openai_client.embeddings.create.assert_called_once()
        call_kwargs = mock_openai_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["input"] == texts
        assert call_kwargs["encoding_format"] == "float"

    def test_encode_empty_list(self, mock_openai_client):
        """Проверяет обработку пустого списка"""
        empty_response = MagicMock(data=[])
        mock_openai_client.embeddings.create.return_value = empty_response

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        result = embedder.encode([])

        assert result == []

    def test_encode_single_text(self, mock_openai_client):
        """Проверяет кодирование одного текста"""
        single_response = MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])
        mock_openai_client.embeddings.create.return_value = single_response

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        result = embedder.encode(["single text"])

        assert len(result) == 1
        assert len(result[0]) == 3

    def test_encode_preserves_original_vectors_when_not_normalized(self, mock_openai_client):
        """Проверяет сохранение исходных векторов без нормализации"""
        original_vectors = [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
        mock_response = MagicMock(
            data=[
                MagicMock(embedding=original_vectors[0]),
                MagicMock(embedding=original_vectors[1]),
            ]
        )
        mock_openai_client.embeddings.create.return_value = mock_response

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            normalize=False,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        texts = ["test text 1", "test text 2"]
        result = embedder.encode(texts)

        for i, vector in enumerate(result):
            assert vector == original_vectors[i]


class TestEncodeQuery:
    """Тесты метода encode_query"""

    def test_encode_query_with_normalize(self, mock_openai_client, fake_response_query):
        """Проверяет кодирование запроса с нормализацией"""
        mock_openai_client.embeddings.create.return_value = fake_response_query

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        result = embedder.encode_query("test query")

        assert len(result) == 3
        # Проверяем нормализацию
        norm = np.linalg.norm(result)
        assert pytest.approx(norm, abs=1e-6) == 1.0

    def test_encode_query_without_normalize(self, mock_openai_client, fake_response_query):
        """Проверяет кодирование запроса без нормализации"""
        mock_openai_client.embeddings.create.return_value = fake_response_query

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            normalize=False,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        result = embedder.encode_query("test query")

        assert len(result) == 3
        # Проверяем, что нормализация не применялась
        assert pytest.approx(np.linalg.norm(result), abs=1e-06) != 1.0

    def test_encode_query_calls_api_with_correct_params(
        self, mock_openai_client, fake_response_query
    ):
        """Проверяет вызов API с правильными параметрами"""
        mock_openai_client.embeddings.create.return_value = fake_response_query

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        embedder.encode_query("test query")

        mock_openai_client.embeddings.create.assert_called_once()
        call_kwargs = mock_openai_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        prefix = load_embedding_query_instruction(_DEFAULT_QUERY_INSTRUCTION_PATH)
        expected = f"{prefix} test query" if prefix else "test query"
        assert call_kwargs["input"] == expected
        assert call_kwargs["encoding_format"] == "float"

    def test_encode_query_skips_instruction_when_disabled(
        self, mock_openai_client, fake_response_query
    ):
        """При use_query_instruction=False в API уходит только исходный запрос."""
        mock_openai_client.embeddings.create.return_value = fake_response_query

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            use_query_instruction=False,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        embedder.encode_query("test query")

        call_kwargs = mock_openai_client.embeddings.create.call_args[1]
        assert call_kwargs["input"] == "test query"

    def test_encode_query_single_embedding_returned(self, mock_openai_client):
        """Проверяет возвращение одного вектора"""
        mock_response = MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])
        mock_openai_client.embeddings.create.return_value = mock_response

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        result = embedder.encode_query("test query")

        assert isinstance(result, list)
        assert len(result) == 3

    def test_encode_query_preserves_original_when_not_normalized(self, mock_openai_client):
        """Проверяет сохранение исходного вектора без нормализации"""
        original_vector = [0.5, 0.5, 0.5]
        mock_response = MagicMock(data=[MagicMock(embedding=original_vector)])
        mock_openai_client.embeddings.create.return_value = mock_response

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            normalize=False,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        result = embedder.encode_query("test query")

        assert result == original_vector

    def test_encode_query_without_instruction_path_sends_raw_query(
        self, mock_openai_client, fake_response_query
    ) -> None:
        """Если путь к файлу не задан, в API уходит только текст запроса."""
        mock_openai_client.embeddings.create.return_value = fake_response_query

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=None,
            use_query_instruction=True,
        )
        embedder.client = mock_openai_client

        embedder.encode_query("мой запрос")

        call_kwargs = mock_openai_client.embeddings.create.call_args[1]
        assert call_kwargs["input"] == "мой запрос"

    def test_encode_query_whitespace_only_instruction_file_sends_raw_query(
        self, mock_openai_client, fake_response_query, tmp_path: Path
    ) -> None:
        """После strip файл только из пробелов — без префикса (как пустая инструкция)."""
        instruction_file = tmp_path / "empty_instr.txt"
        instruction_file.write_text("   \n\n  ", encoding="utf-8")

        mock_openai_client.embeddings.create.return_value = fake_response_query

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=instruction_file,
            use_query_instruction=True,
        )
        embedder.client = mock_openai_client

        embedder.encode_query("query")

        call_kwargs = mock_openai_client.embeddings.create.call_args[1]
        assert call_kwargs["input"] == "query"


class TestEmbedderRetries:
    """Ретраи через retry_sync при временных ошибках embeddings API."""

    @patch("cadence_md.app.retry_utils.sleep_with_backoff")
    def test_encode_retries_on_rate_limit(
        self, _mock_sleep: MagicMock, mock_openai_client, fake_response_data
    ) -> None:
        err = RateLimitError("429", response=MagicMock(), body=None)
        mock_openai_client.embeddings.create.side_effect = [err, fake_response_data]

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            max_retries=3,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        result = embedder.encode(["x", "y"])

        assert len(result) == 2
        assert mock_openai_client.embeddings.create.call_count == 2

    @patch("cadence_md.app.retry_utils.sleep_with_backoff")
    def test_encode_query_retries_on_rate_limit(
        self, _mock_sleep: MagicMock, mock_openai_client, fake_response_query
    ) -> None:
        err = RateLimitError("429", response=MagicMock(), body=None)
        mock_openai_client.embeddings.create.side_effect = [err, fake_response_query]

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            max_retries=4,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        out = embedder.encode_query("q")

        assert len(out) == 3
        assert mock_openai_client.embeddings.create.call_count == 2

    @patch("cadence_md.app.retry_utils.sleep_with_backoff")
    def test_encode_exhausts_retries_on_persistent_rate_limit(
        self, _mock_sleep: MagicMock, mock_openai_client
    ) -> None:
        err = RateLimitError("429", response=MagicMock(), body=None)
        mock_openai_client.embeddings.create.side_effect = err

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            max_retries=2,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        with pytest.raises(RateLimitError):
            embedder.encode(["a"])

        assert mock_openai_client.embeddings.create.call_count == 2

    def test_encode_no_retry_on_non_retryable_error(self, mock_openai_client) -> None:
        mock_openai_client.embeddings.create.side_effect = ValueError("bad input")

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            max_retries=5,
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        with pytest.raises(ValueError, match="bad input"):
            embedder.encode(["a"])

        assert mock_openai_client.embeddings.create.call_count == 1


class TestL2Normalize:
    """Тесты метода _l2_normalize"""

    def test_l2_normalize_single_vector(self, mock_openai_client):
        """Проверяет нормализацию одного вектора"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        vector = np.array([3.0, 4.0])
        result = embedder._l2_normalize(vector)

        # Проверяем, что норма ≈ 1
        norm = np.linalg.norm(result)
        assert pytest.approx(norm, abs=1e-6) == 1.0

        # Проверяем соотношения (3-4-5 треугольник)
        assert pytest.approx(result[0], abs=1e-6) == 3.0 / 5.0
        assert pytest.approx(result[1], abs=1e-6) == 4.0 / 5.0

    def test_l2_normalize_multiple_vectors(self, mock_openai_client):
        """Проверяет нормализацию матрицы векторов"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        vectors = np.array(
            [
                [3.0, 4.0],
                [5.0, 12.0],
            ]
        )
        result = embedder._l2_normalize(vectors)

        assert result.shape == (2, 2)
        for vector in result:
            norm = np.linalg.norm(vector)
            assert pytest.approx(norm, abs=1e-6) == 1.0

    def test_l2_normalize_with_zero_vector(self, mock_openai_client):
        """Проверяет обработку нулевого вектора (без деления на 0)"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        zero_vector = np.array([0.0, 0.0, 0.0])
        result = embedder._l2_normalize(zero_vector)

        # После добавления 1e-12 норма будет ~1e-12
        norm = np.linalg.norm(result)
        assert pytest.approx(norm, abs=1e-6) == 1e-12

        # Все компоненты должны быть 0
        assert np.allclose(result, 0.0)

    def test_l2_normalize_already_normalized(self, mock_openai_client):
        """Проверяет уже нормализованного вектора"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        normalized_vector = np.array([1.0, 0.0, 0.0])
        result = embedder._l2_normalize(normalized_vector)

        assert pytest.approx(result[0], abs=1e-6) == 1.0
        assert pytest.approx(result[1], abs=1e-6) == 0.0
        assert pytest.approx(result[2], abs=1e-6) == 0.0

    def test_l2_normalize_returns_correct_dtype(self, mock_openai_client):
        """Проверяет тип возвращаемого массива"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        vector = np.array([3.0, 4.0], dtype="float32")
        result = embedder._l2_normalize(vector)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_l2_normalize_negative_components(self, mock_openai_client):
        """Проверяет нормализацию вектора с отрицательными компонентами"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        vector = np.array([-3.0, -4.0])
        result = embedder._l2_normalize(vector)

        norm = np.linalg.norm(result)
        assert pytest.approx(norm, abs=1e-6) == 1.0

        # Соотнощения сохранены
        assert pytest.approx(result[0], abs=1e-6) == -3.0 / 5.0
        assert pytest.approx(result[1], abs=1e-6) == -4.0 / 5.0

    def test_l2_normalize_3d_vector(self, mock_openai_client):
        """Проверяет нормализацию 3D вектора"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        vector = np.array([1.0, 2.0, 3.0])
        result = embedder._l2_normalize(vector)

        norm = np.linalg.norm(result)
        assert pytest.approx(norm, abs=1e-6) == 1.0

    def test_l2_normalize_matrix_of_vectors(self, mock_openai_client):
        """Проверяет нормализацию матрицы (batch processing)"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        vectors = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        result = embedder._l2_normalize(vectors)

        assert result.shape == (3, 3)
        for vector in result:
            norm = np.linalg.norm(vector)
            assert pytest.approx(norm, abs=1e-6) == 1.0

    def test_l2_normalize_matrix_with_zero_row(self, mock_openai_client):
        """Нулевая строка в batch: остаётся нулевой, остальные строки — единичная норма."""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        vectors = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
            ]
        )
        result = embedder._l2_normalize(vectors)

        assert np.allclose(result[0], 0.0)
        assert pytest.approx(np.linalg.norm(result[1]), abs=1e-6) == 1.0

    def test_l2_normalize_preserves_relative_directions(self, mock_openai_client):
        """Проверяет сохранение относительных направлений после нормализации"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        vectors = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],  # То же направление
            ]
        )
        result = embedder._l2_normalize(vectors)

        # Проверяем колинеарность
        cos_sim = np.dot(result[0], result[1])
        assert pytest.approx(cos_sim, abs=1e-6) == 1.0


class TestGetEmbedder:
    """Тесты функции get_embedder"""

    def test_get_embedder_returns_correct_type(self, mock_openai_client, settings):
        """Проверяет тип возвращаемого объекта"""

        result = get_embedder(
            model=settings.rag_config.embedding.model_name,
            normalize=settings.rag_config.embedding.normalize_embeddings,
            return_score=settings.rag_config.embedding.return_score,
            base_url=settings.MODEL_INFERENCE_BASE_URL,
            api_key=settings.MODEL_INFERENCE_API_KEY,
            use_query_instruction=settings.rag_config.embedding.use_query_instruction,
            query_instruction_path=settings.rag_config.embedding.query_instruction_path,
        )

        assert isinstance(result, EmbedderWrapper)

    def test_get_embedder_uses_settings(self, mock_openai_client, settings):
        """Проверяет использование настроек"""

        embedder = get_embedder(
            model=settings.rag_config.embedding.model_name,
            normalize=settings.rag_config.embedding.normalize_embeddings,
            return_score=settings.rag_config.embedding.return_score,
            base_url=settings.MODEL_INFERENCE_BASE_URL,
            api_key=settings.MODEL_INFERENCE_API_KEY,
            use_query_instruction=settings.rag_config.embedding.use_query_instruction,
            query_instruction_path=settings.rag_config.embedding.query_instruction_path,
        )

        assert embedder.model == settings.rag_config.embedding.model_name
        assert embedder.normalize == settings.rag_config.embedding.normalize_embeddings
        assert embedder.return_score == settings.rag_config.embedding.return_score
        assert embedder.use_query_instruction == settings.rag_config.embedding.use_query_instruction

    def test_get_embedder_forwards_http_and_retry_options(self, mock_openai_client) -> None:
        embedder = get_embedder(
            model="m",
            normalize=True,
            return_score=False,
            base_url="http://embed:9999/v1",
            api_key="secret",
            query_instruction_path=None,
            timeout_seconds=42.5,
            max_retries=2,
            backoff_base_seconds=1.25,
            backoff_max_seconds=30.0,
        )

        assert embedder.model == "m"
        assert embedder.client.timeout == 42.5
        assert embedder._max_retries == 2
        assert embedder._backoff_base_seconds == 1.25
        assert embedder._backoff_max_seconds == 30.0

    def test_get_embedder_from_settings(self, mock_openai_client, settings):
        """Проверяет builder из глобальных настроек приложения."""
        embedder = get_embedder_from_settings(settings)

        assert isinstance(embedder, EmbedderWrapper)
        assert embedder.model == settings.rag_config.embedding.model_name
        assert embedder.normalize == settings.rag_config.embedding.normalize_embeddings
        assert embedder.return_score == settings.rag_config.embedding.return_score

    def test_get_embedder_from_settings_accepts_magic_mock_settings(
        self, mock_openai_client
    ) -> None:
        """Явно переданный объект настроек (например MagicMock) пробрасывается в get_embedder."""
        mock_app_settings = MagicMock()
        mock_app_settings.MODEL_INFERENCE_BASE_URL = "http://mock-inference/v1"
        mock_app_settings.MODEL_INFERENCE_API_KEY = "mock-key"
        emb_cfg = MagicMock()
        emb_cfg.model_name = "model-from-mock"
        emb_cfg.normalize_embeddings = False
        emb_cfg.return_score = True
        emb_cfg.use_query_instruction = False
        emb_cfg.query_instruction_path = None
        emb_cfg.timeout_seconds = 11.0
        emb_cfg.max_retries = 2
        emb_cfg.backoff_base_seconds = 1.0
        emb_cfg.backoff_max_seconds = 2.0
        mock_app_settings.rag_config.embedding = emb_cfg

        embedder = get_embedder_from_settings(mock_app_settings)

        assert embedder.model == "model-from-mock"
        assert embedder.normalize is False
        assert embedder.return_score is True
        assert embedder.use_query_instruction is False
        assert embedder.client.timeout == 11.0


class TestEmbedderRetrySyncWiring:
    """Проверка имён операций и параметров retry_sync в обёртке."""

    @patch("cadence_md.app.embedder.retry_sync")
    def test_encode_passes_operation_name_to_retry_sync(
        self, mock_retry_sync: MagicMock, mock_openai_client, fake_response_data
    ) -> None:
        mock_retry_sync.return_value = fake_response_data

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        embedder.encode(["a"])

        mock_retry_sync.assert_called_once()
        kwargs = mock_retry_sync.call_args[1]
        assert kwargs["operation_name"] == "embeddings.encode"
        assert kwargs["max_attempts"] == embedder._max_retries

    @patch("cadence_md.app.embedder.retry_sync")
    def test_encode_query_passes_operation_name_to_retry_sync(
        self, mock_retry_sync: MagicMock, mock_openai_client, fake_response_query
    ) -> None:
        mock_retry_sync.return_value = fake_response_query

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
            query_instruction_path=_DEFAULT_QUERY_INSTRUCTION_PATH,
        )
        embedder.client = mock_openai_client

        embedder.encode_query("q")

        mock_retry_sync.assert_called_once()
        kwargs = mock_retry_sync.call_args[1]
        assert kwargs["operation_name"] == "embeddings.encode_query"
