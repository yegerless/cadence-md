import sys
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

sys.path.insert(0, "..")

from cadence_md.app.enums import RerankerAggregationStrategy
from cadence_md.app.reranker import (
    RerankerWrapper,
    aggregate_embedding,
    get_reranker,
)


class TestAggregateEmbedding:
    """Тесты aggregate_embedding и веток агрегации."""

    def test_l2_norm_strategy(self) -> None:
        assert aggregate_embedding(
            [3.0, 4.0], RerankerAggregationStrategy.L2_NORM
        ) == pytest.approx(5.0)

    def test_mean_strategy(self) -> None:
        assert aggregate_embedding([3.0, 4.0], RerankerAggregationStrategy.MEAN) == pytest.approx(
            3.5
        )

    def test_max_strategy(self) -> None:
        assert aggregate_embedding([3.0, 4.0], RerankerAggregationStrategy.MAX) == pytest.approx(
            4.0
        )

    def test_empty_vector_mean_uses_safe_vec(self) -> None:
        """Пустой iterable даёт [0.0], mean = 0.0."""
        assert aggregate_embedding([], RerankerAggregationStrategy.MEAN) == pytest.approx(0.0)

    def test_empty_vector_l2_norm(self) -> None:
        assert aggregate_embedding([], RerankerAggregationStrategy.L2_NORM) == pytest.approx(0.0)

    def test_empty_vector_max_abs(self) -> None:
        assert aggregate_embedding([], RerankerAggregationStrategy.MAX) == pytest.approx(0.0)

    def test_unknown_strategy_falls_back_to_l2(self) -> None:
        """Нестандартное значение strategy — fallback на L2."""
        bogus = cast(Any, "not_a_listed_strategy")
        assert aggregate_embedding([3.0, 4.0], bogus) == pytest.approx(5.0)


class TestRerankerWrapperStaticHelpers:
    """Статические хелперы форматирования."""

    def test_normalize_instruction_none(self) -> None:
        assert RerankerWrapper._normalize_instruction(None) is None

    def test_normalize_instruction_whitespace_only(self) -> None:
        assert RerankerWrapper._normalize_instruction("  \n  ") is None

    def test_normalize_instruction_strips_and_keeps_content(self) -> None:
        assert RerankerWrapper._normalize_instruction("  hello  ") == "hello"

    def test_query_passage_body(self) -> None:
        body = RerankerWrapper._query_passage_body("q1", "p1")
        assert body == "query: q1\npassage: p1"

    def test_format_rerank_input_without_instruction(self, mock_openai_client: MagicMock) -> None:
        reranker = RerankerWrapper(
            model="m",
            instruction=None,
            top_k=10,
            return_score=True,
            embedding_agregation_strategy=RerankerAggregationStrategy.L2_NORM,
            base_url="http://localhost",
            api_key="k",
        )
        reranker.client = mock_openai_client
        assert reranker._format_rerank_input("q", "p") == "query: q\npassage: p"

    def test_format_rerank_input_with_instruction(self, mock_openai_client: MagicMock) -> None:
        reranker = RerankerWrapper(
            model="m",
            instruction="  Instruct me  ",
            top_k=10,
            return_score=True,
            embedding_agregation_strategy=RerankerAggregationStrategy.L2_NORM,
            base_url="http://localhost",
            api_key="k",
        )
        reranker.client = mock_openai_client
        expected = "Instruct me\nquery: q\npassage: p"
        assert reranker._format_rerank_input("q", "p") == expected

    def test_build_inputs(self, mock_openai_client: MagicMock) -> None:
        reranker = RerankerWrapper(
            model="m",
            instruction="prefix",
            top_k=10,
            return_score=True,
            embedding_agregation_strategy=RerankerAggregationStrategy.L2_NORM,
            base_url="http://localhost",
            api_key="k",
        )
        reranker.client = mock_openai_client
        docs = [Document(page_content="a"), Document(page_content="b")]
        inputs = reranker._build_inputs("q", docs)
        assert inputs == [
            "prefix\nquery: q\npassage: a",
            "prefix\nquery: q\npassage: b",
        ]


class TestEncodePairs:
    """Тесты encode_pairs с моком клиента."""

    def _make_reranker(
        self,
        mock_openai_client: MagicMock,
        strategy: RerankerAggregationStrategy,
    ) -> RerankerWrapper:
        reranker = RerankerWrapper(
            model="rerank-model",
            instruction=None,
            top_k=10,
            return_score=True,
            embedding_agregation_strategy=strategy,
            base_url="http://localhost",
            api_key="k",
        )
        reranker.client = mock_openai_client
        return reranker

    def test_encode_pairs_calls_api_and_returns_scores(self, mock_openai_client: MagicMock) -> None:
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[3.0, 4.0]),
                MagicMock(embedding=[0.0, 0.0, 2.0]),
            ]
        )
        reranker = self._make_reranker(mock_openai_client, RerankerAggregationStrategy.L2_NORM)
        docs = [Document(page_content="p1"), Document(page_content="p2")]
        scores = reranker.encode_pairs("query", docs)

        assert len(scores) == 2
        assert scores[0] == pytest.approx(5.0)
        assert scores[1] == pytest.approx(2.0)

        mock_openai_client.embeddings.create.assert_called_once()
        kwargs = mock_openai_client.embeddings.create.call_args[1]
        assert kwargs["model"] == "rerank-model"
        assert kwargs["encoding_format"] == "float"
        assert kwargs["input"] == [
            "query: query\npassage: p1",
            "query: query\npassage: p2",
        ]

    def test_encode_pairs_mean_strategy_differs_from_l2(
        self, mock_openai_client: MagicMock
    ) -> None:
        emb = [2.0, 2.0, 2.0]
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=emb)]
        )
        reranker = self._make_reranker(mock_openai_client, RerankerAggregationStrategy.MEAN)
        scores = reranker.encode_pairs("q", [Document(page_content="x")])
        assert scores[0] == pytest.approx(2.0)

        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=emb)]
        )
        reranker_l2 = self._make_reranker(mock_openai_client, RerankerAggregationStrategy.L2_NORM)
        scores_l2 = reranker_l2.encode_pairs("q", [Document(page_content="x")])
        assert scores_l2[0] == pytest.approx(12.0**0.5)


class TestRerank:
    """Тесты rerank: порядок, top_k, return_score, пустой список."""

    def test_rerank_empty_documents(self, mock_openai_client: MagicMock) -> None:
        reranker = RerankerWrapper(
            model="m",
            instruction=None,
            top_k=10,
            return_score=True,
            embedding_agregation_strategy=RerankerAggregationStrategy.L2_NORM,
            base_url="http://localhost",
            api_key="k",
        )
        reranker.client = mock_openai_client
        assert reranker.rerank("q", []) == []

    def test_rerank_sorts_by_score_descending(self, mock_openai_client: MagicMock) -> None:
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[1.0, 0.0, 0.0]),
                MagicMock(embedding=[3.0, 4.0, 0.0]),
            ]
        )
        reranker = RerankerWrapper(
            model="m",
            instruction=None,
            top_k=10,
            return_score=True,
            embedding_agregation_strategy=RerankerAggregationStrategy.L2_NORM,
            base_url="http://localhost",
            api_key="k",
        )
        reranker.client = mock_openai_client
        low = Document(page_content="low")
        high = Document(page_content="high")
        pairs = reranker.rerank("q", [low, high])
        assert [d.page_content for d, _ in pairs] == ["high", "low"]
        assert pairs[0][1] == pytest.approx(5.0)
        assert pairs[1][1] == pytest.approx(1.0)

    def test_rerank_top_k_truncates(self, mock_openai_client: MagicMock) -> None:
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[1.0, 0.0]),
                MagicMock(embedding=[0.0, 1.0]),
                MagicMock(embedding=[3.0, 4.0]),
            ]
        )
        reranker = RerankerWrapper(
            model="m",
            instruction=None,
            top_k=2,
            return_score=True,
            embedding_agregation_strategy=RerankerAggregationStrategy.L2_NORM,
            base_url="http://localhost",
            api_key="k",
        )
        reranker.client = mock_openai_client
        docs = [
            Document(page_content="a"),
            Document(page_content="b"),
            Document(page_content="c"),
        ]
        pairs = reranker.rerank("q", docs)
        assert len(pairs) == 2
        assert pairs[0][1] >= pairs[1][1]

    def test_rerank_return_score_false(self, mock_openai_client: MagicMock) -> None:
        mock_openai_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[0.0, 1.0]),
                MagicMock(embedding=[3.0, 4.0]),
            ]
        )
        reranker = RerankerWrapper(
            model="m",
            instruction=None,
            top_k=10,
            return_score=False,
            embedding_agregation_strategy=RerankerAggregationStrategy.L2_NORM,
            base_url="http://localhost",
            api_key="k",
        )
        reranker.client = mock_openai_client
        d1 = Document(page_content="first_by_score")
        d2 = Document(page_content="second")
        pairs = reranker.rerank("q", [d1, d2])
        assert all(score is None for _, score in pairs)
        assert pairs[0][0].page_content == "second"


class TestGetReranker:
    """Фабрика get_reranker."""

    def test_get_reranker_returns_configured_wrapper(self, mock_openai_client: MagicMock) -> None:
        wrapper = get_reranker(
            model="rm",
            instruction="  hi  ",
            top_k=3,
            return_score=False,
            embedding_agregation_strategy=RerankerAggregationStrategy.MAX,
            base_url="http://api",
            api_key="key",
        )
        wrapper.client = mock_openai_client
        assert isinstance(wrapper, RerankerWrapper)
        assert wrapper.model == "rm"
        assert wrapper.instruction == "  hi  "
        assert wrapper._instruction_for_prompt == "hi"
        assert wrapper.top_k == 3
        assert wrapper.return_score is False
        assert wrapper.embedding_agregation_strategy == RerankerAggregationStrategy.MAX
