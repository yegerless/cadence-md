import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, "..")

from cadence_md.app.embedder import EmbedderWrapper, get_embedder


class TestEmbedderWrapperInit:
    """Тесты инициализации EmbedderWrapper"""

    def test_init_basic(self, mock_openai_client):
        """Проверяет инициализацию с минимальными параметрами"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
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
        )
        assert embedder.normalize is True

    def test_init_default_return_score_false(self, mock_openai_client):
        """Проверяет, что return_score по умолчанию False"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
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
        )
        embedder.client = mock_openai_client

        embedder.encode_query("test query")

        mock_openai_client.embeddings.create.assert_called_once()
        call_kwargs = mock_openai_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["input"] == "test query"
        assert call_kwargs["encoding_format"] == "float"

    def test_encode_query_single_embedding_returned(self, mock_openai_client):
        """Проверяет возвращение одного вектора"""
        mock_response = MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])
        mock_openai_client.embeddings.create.return_value = mock_response

        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
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
        )
        embedder.client = mock_openai_client

        result = embedder.encode_query("test query")

        assert result == original_vector


class TestL2Normalize:
    """Тесты метода _l2_normalize"""

    def test_l2_normalize_single_vector(self, mock_openai_client):
        """Проверяет нормализацию одного вектора"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
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

    def test_l2_normalize_preserves_relative_directions(self, mock_openai_client):
        """Проверяет сохранение относительных направлений после нормализации"""
        embedder = EmbedderWrapper(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234",
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

    def test_get_embedder_returns_correct_type(self, mock_openai_client):
        """Проверяет тип возвращаемого объекта"""

        result = get_embedder()

        assert isinstance(result, EmbedderWrapper)

    def test_get_embedder_uses_settings(self, mock_openai_client):
        """Проверяет использование настроек"""

        embedder = get_embedder()

        assert embedder.model == "text-embedding-bge-m3"
        assert embedder.normalize is True
        assert embedder.return_score is False
