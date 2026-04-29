import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, "..")

from cadence_md.app.reranker import (
    RerankerAPIError,
    RerankerWrapper,
    _parse_rerank_response,
    _rerank_url,
    get_reranker,
)


class TestRerankUrl:
    """``_rerank_url`` builds ``/v1/rerank`` from common base URL shapes."""

    def test_with_v1_suffix(self) -> None:
        assert _rerank_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1/rerank"
        assert _rerank_url("http://127.0.0.1:1234/v1/") == "http://127.0.0.1:1234/v1/rerank"

    def test_without_v1_suffix(self) -> None:
        assert _rerank_url("http://127.0.0.1:1234") == "http://127.0.0.1:1234/v1/rerank"


class TestParseRerankResponse:
    """``_parse_rerank_response`` maps ``results`` to ordered scores."""

    def test_ordered_scores(self) -> None:
        data = {
            "results": [
                {"index": 1, "relevance_score": 2.5},
                {"index": 0, "relevance_score": 1.0},
            ]
        }
        assert _parse_rerank_response(data, expected_documents=2) == [1.0, 2.5]

    def test_raises_on_missing_index(self) -> None:
        data = {"results": [{"index": 0, "relevance_score": 1.0}]}
        with pytest.raises(RerankerAPIError, match="Incomplete rerank results"):
            _parse_rerank_response(data, expected_documents=2)

    def test_raises_on_bad_results_type(self) -> None:
        with pytest.raises(RerankerAPIError, match="missing 'results'"):
            _parse_rerank_response({"results": None}, expected_documents=1)

    def test_raises_on_index_out_of_range(self) -> None:
        data = {"results": [{"index": 2, "relevance_score": 1.0}]}
        with pytest.raises(RerankerAPIError, match="out of range"):
            _parse_rerank_response(data, expected_documents=2)


def _reranker_with_stub_post(
    stub: MagicMock,
    *,
    top_k: int = 10,
    return_score: bool = True,
) -> RerankerWrapper:
    r = RerankerWrapper(
        model="rerank-model",
        top_k=top_k,
        return_score=return_score,
        base_url="http://localhost:1234/v1",
        api_key="k",
        timeout_s=30.0,
        max_retries_on_rate_limit=2,
    )
    r._post_rerank = stub  # type: ignore[method-assign]
    return r


class TestScorePairs:
    """``score_pairs`` calls ``_post_rerank`` with query and document texts."""

    def test_score_pairs_payload_and_order(self) -> None:
        stub = MagicMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": -1.0},
                    {"index": 1, "relevance_score": 5.0},
                ]
            }
        )
        reranker = _reranker_with_stub_post(stub)
        docs = [Document(page_content="a"), Document(page_content="b")]
        scores = reranker.score_pairs("my query", docs)

        assert scores == [-1.0, 5.0]
        stub.assert_called_once()
        payload = stub.call_args[0][0]
        assert payload["model"] == "rerank-model"
        assert payload["query"] == "my query"
        assert payload["documents"] == ["a", "b"]

    def test_score_pairs_empty(self) -> None:
        stub = MagicMock()
        reranker = _reranker_with_stub_post(stub)
        assert reranker.score_pairs("q", []) == []
        stub.assert_not_called()


class TestRerank:
    """``rerank``: sorting, ``top_k``, ``return_score``, empty input."""

    def test_rerank_empty_documents(self) -> None:
        stub = MagicMock()
        reranker = _reranker_with_stub_post(stub)
        assert reranker.rerank("q", []) == []
        stub.assert_not_called()

    def test_rerank_sorts_by_score_descending(self) -> None:
        stub = MagicMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                    {"index": 1, "relevance_score": 9.0},
                ]
            }
        )
        reranker = _reranker_with_stub_post(stub)
        low = Document(page_content="low")
        high = Document(page_content="high")
        pairs = reranker.rerank("q", [low, high])

        assert [d.page_content for d, _ in pairs] == ["high", "low"]
        assert pairs[0][1] == pytest.approx(9.0)
        assert pairs[1][1] == pytest.approx(1.0)

    def test_rerank_top_k_truncates(self) -> None:
        stub = MagicMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                    {"index": 1, "relevance_score": 2.0},
                    {"index": 2, "relevance_score": 3.0},
                ]
            }
        )
        reranker = _reranker_with_stub_post(stub, top_k=2)
        docs = [
            Document(page_content="a"),
            Document(page_content="b"),
            Document(page_content="c"),
        ]
        pairs = reranker.rerank("q", docs)
        assert len(pairs) == 2
        assert pairs[0][1] >= pairs[1][1]

    def test_rerank_return_score_false(self) -> None:
        stub = MagicMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                    {"index": 1, "relevance_score": 10.0},
                ]
            }
        )
        reranker = _reranker_with_stub_post(stub, return_score=False)
        d1 = Document(page_content="first")
        d2 = Document(page_content="second")
        pairs = reranker.rerank("q", [d1, d2])
        assert all(s is None for _, s in pairs)
        assert pairs[0][0].page_content == "second"


class TestPostRerankHTTP:
    """HTTP layer: success path and error handling."""

    def _client_context(self, post_return: MagicMock) -> tuple[MagicMock, MagicMock]:
        """Return (context_manager, inner_client) for ``with httpx.Client(...)``."""
        inner = MagicMock()
        inner.post.return_value = post_return
        cm = MagicMock()
        cm.__enter__.return_value = inner
        cm.__exit__.return_value = None
        return cm, inner

    def test_post_rerank_success(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.text = ""
        resp.json.return_value = {"results": [{"index": 0, "relevance_score": 3.14}]}
        cm, inner = self._client_context(resp)

        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="secret",
            max_retries_on_rate_limit=0,
        )
        with patch("cadence_md.app.reranker.httpx.Client", return_value=cm):
            out = reranker._post_rerank({"model": "m", "query": "q", "documents": ["x"]})

        assert out == {"results": [{"index": 0, "relevance_score": 3.14}]}
        inner.post.assert_called_once()
        _args, kwargs = inner.post.call_args
        assert kwargs["json"]["query"] == "q"
        assert kwargs["headers"]["Authorization"] == "Bearer secret"

    def test_post_rerank_http_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "boom"
        cm, _inner = self._client_context(resp)
        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=0,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=cm),
            pytest.raises(RerankerAPIError, match="Rerank HTTP 500"),
        ):
            reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

    def test_post_rerank_429_then_success(self) -> None:
        fail = MagicMock(status_code=429, text="")
        ok = MagicMock(status_code=200, text="")
        ok.json.return_value = {"results": [{"index": 0, "relevance_score": 1.0}]}
        mock_client = MagicMock()
        mock_client.post.side_effect = [fail, ok]
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_client
        mock_cm.__exit__.return_value = None

        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=2,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=mock_cm),
            patch("cadence_md.app.reranker.time.sleep", MagicMock()),
        ):
            out = reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

        assert out["results"][0]["relevance_score"] == 1.0
        assert mock_client.post.call_count == 2


class TestGetReranker:
    """Factory ``get_reranker``."""

    def test_get_reranker_returns_configured_wrapper(self) -> None:
        wrapper = get_reranker(
            model="rm",
            top_k=3,
            return_score=False,
            base_url="http://api/v1",
            api_key="key",
            timeout_s=60.0,
            max_retries_on_rate_limit=5,
        )
        assert isinstance(wrapper, RerankerWrapper)
        assert wrapper.model == "rm"
        assert wrapper.top_k == 3
        assert wrapper.return_score is False
        assert wrapper._timeout_s == 60.0
        assert wrapper._max_retries_on_rate_limit == 5
        assert wrapper._api_url == "http://api/v1/rerank"
