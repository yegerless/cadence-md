import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.documents import Document

sys.path.insert(0, "..")

from cadence_md.app.reranker import (
    RerankerAPIError,
    RerankerWrapper,
    _parse_rerank_response,
    _rerank_url,
    get_reranker,
    get_reranker_from_settings,
    load_reranker_query_instruction,
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

    def test_skips_non_mapping_result_entries(self) -> None:
        data = {
            "results": [
                "ignored",
                {"index": 1, "relevance_score": 2.0},
                {"index": 0, "relevance_score": 1.0},
            ]
        }
        assert _parse_rerank_response(data, expected_documents=2) == [1.0, 2.0]

    def test_raises_on_invalid_result_entry(self) -> None:
        data = {"results": [{"index": "not_int", "relevance_score": 1.0}]}
        with pytest.raises(RerankerAPIError, match="Invalid rerank result entry"):
            _parse_rerank_response(data, expected_documents=1)

    def test_raises_when_results_key_missing(self) -> None:
        with pytest.raises(RerankerAPIError, match="missing 'results'"):
            _parse_rerank_response({}, expected_documents=1)


def _reranker_with_stub_post(
    stub: MagicMock,
    *,
    top_k: int = 10,
    return_score: bool = True,
    use_query_instruction: bool = False,
    query_instruction_path: Path | None = None,
) -> RerankerWrapper:
    r = RerankerWrapper(
        model="rerank-model",
        top_k=top_k,
        return_score=return_score,
        base_url="http://localhost:1234/v1",
        api_key="k",
        timeout_s=30.0,
        max_retries_on_rate_limit=2,
        use_query_instruction=use_query_instruction,
        query_instruction_path=query_instruction_path,
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

    def test_score_pairs_uses_document_texts_when_provided(self) -> None:
        stub = MagicMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                ]
            }
        )
        reranker = _reranker_with_stub_post(stub)
        docs = [Document(page_content="short")]
        reranker.score_pairs("q", docs, document_texts=["long rerank text"])
        payload = stub.call_args[0][0]
        assert payload["documents"] == ["long rerank text"]

    def test_score_pairs_prepends_instruction_when_enabled(self, tmp_path: Path) -> None:
        instr_file = tmp_path / "instr.txt"
        instr_file.write_text("CUSTOM_RERANK_TASK\n", encoding="utf-8")
        stub = MagicMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                ]
            }
        )
        reranker = _reranker_with_stub_post(
            stub,
            use_query_instruction=True,
            query_instruction_path=instr_file,
        )
        reranker.score_pairs("пользовательский запрос", [Document(page_content="x")])
        payload = stub.call_args[0][0]
        assert payload["query"].startswith("CUSTOM_RERANK_TASK ")
        assert payload["query"].endswith("пользовательский запрос")

    def test_score_pairs_raw_query_when_instruction_disabled(self) -> None:
        stub = MagicMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                ]
            }
        )
        reranker = _reranker_with_stub_post(stub, use_query_instruction=False)
        reranker.score_pairs("only this", [Document(page_content="x")])
        payload = stub.call_args[0][0]
        assert payload["query"] == "only this"

    def test_score_pairs_no_instruction_file_does_not_prepend(self) -> None:
        """``use_query_instruction`` alone does not change query without a loaded file."""
        stub = MagicMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                ]
            }
        )
        reranker = _reranker_with_stub_post(stub, use_query_instruction=True)
        reranker.score_pairs("plain", [Document(page_content="x")])
        assert stub.call_args[0][0]["query"] == "plain"

    def test_score_pairs_document_texts_length_mismatch(self) -> None:
        stub = MagicMock()
        reranker = _reranker_with_stub_post(stub)
        with pytest.raises(ValueError, match="document_texts length must match"):
            reranker.score_pairs(
                "q",
                [Document(page_content="a"), Document(page_content="b")],
                document_texts=["only_one"],
            )
        stub.assert_not_called()


class TestLoadRerankerQueryInstruction:
    """``load_reranker_query_instruction`` strips file edges; keeps inner newlines."""

    def test_strips_edges_preserves_internal_newlines(self, tmp_path: Path) -> None:
        p = tmp_path / "i.txt"
        p.write_text("line one\nline two", encoding="utf-8")
        assert load_reranker_query_instruction(p) == "line one\nline two"


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

    def test_post_rerank_success(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.text = ""
        resp.json.return_value = {"results": [{"index": 0, "relevance_score": 3.14}]}
        inner = MagicMock()
        inner.post.return_value = resp

        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="secret",
            max_retries_on_rate_limit=0,
        )
        with patch("cadence_md.app.reranker.httpx.Client", return_value=inner):
            out = reranker._post_rerank({"model": "m", "query": "q", "documents": ["x"]})

        assert out == {"results": [{"index": 0, "relevance_score": 3.14}]}
        inner.post.assert_called_once()
        _args, kwargs = inner.post.call_args
        assert kwargs["json"]["query"] == "q"
        assert kwargs["headers"]["Authorization"] == "Bearer secret"

    def test_post_rerank_http_error(self) -> None:
        resp = MagicMock()
        resp.status_code = 400
        resp.text = "boom"
        inner = MagicMock()
        inner.post.return_value = resp
        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=0,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=inner),
            pytest.raises(RerankerAPIError, match="Rerank HTTP 400"),
        ):
            reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

    def test_post_rerank_429_then_success(self) -> None:
        fail = MagicMock(status_code=429, text="")
        ok = MagicMock(status_code=200, text="")
        ok.json.return_value = {"results": [{"index": 0, "relevance_score": 1.0}]}
        mock_client = MagicMock()
        mock_client.post.side_effect = [fail, ok]

        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=2,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=mock_client),
            patch("cadence_md.app.retry_utils.time.sleep", MagicMock()),
        ):
            out = reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

        assert out["results"][0]["relevance_score"] == 1.0
        assert mock_client.post.call_count == 2

    def test_post_rerank_500_then_success(self) -> None:
        fail = MagicMock(status_code=500, text="busy")
        ok = MagicMock(status_code=200, text="")
        ok.json.return_value = {"results": [{"index": 0, "relevance_score": 2.0}]}
        mock_client = MagicMock()
        mock_client.post.side_effect = [fail, ok]

        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_transport=2,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=mock_client),
            patch("cadence_md.app.retry_utils.time.sleep", MagicMock()),
        ):
            out = reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

        assert out["results"][0]["relevance_score"] == 2.0
        assert mock_client.post.call_count == 2

    def test_post_rerank_invalid_json_raises(self) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "not json {{{"
        resp.json.side_effect = ValueError("bad json")
        inner = MagicMock()
        inner.post.return_value = resp
        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=0,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=inner),
            pytest.raises(RerankerAPIError, match="not valid JSON"),
        ):
            reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

    def test_post_rerank_transport_error_then_success(self) -> None:
        ok = MagicMock(status_code=200, text="")
        ok.json.return_value = {"results": [{"index": 0, "relevance_score": 7.0}]}
        mock_client = MagicMock()
        mock_client.post.side_effect = [httpx.ConnectError("refused"), ok]

        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_transport=2,
            max_retries_on_rate_limit=0,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=mock_client),
            patch("cadence_md.app.retry_utils.time.sleep", MagicMock()),
        ):
            out = reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

        assert out["results"][0]["relevance_score"] == 7.0
        assert mock_client.post.call_count == 2

    def test_post_rerank_429_no_retries_raises(self) -> None:
        fail = MagicMock(status_code=429, text="limit")
        mock_client = MagicMock()
        mock_client.post.return_value = fail

        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=0,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=mock_client),
            pytest.raises(RerankerAPIError, match="429 too many times"),
        ):
            reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

        mock_client.post.assert_called_once()

    def test_post_rerank_429_exhausts_budget(self) -> None:
        fail = MagicMock(status_code=429, text="")
        mock_client = MagicMock()
        mock_client.post.return_value = fail

        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=1,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=mock_client),
            patch("cadence_md.app.retry_utils.time.sleep", MagicMock()),
            pytest.raises(RerankerAPIError, match="429 too many times"),
        ):
            reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

        assert mock_client.post.call_count == 2

    def test_post_rerank_retryable_status_exhausts_transport_budget(self) -> None:
        fail = MagicMock(status_code=503, text="unavailable")
        mock_client = MagicMock()
        mock_client.post.return_value = fail

        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=0,
            max_retries_on_transport=1,
        )
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=mock_client),
            patch("cadence_md.app.retry_utils.time.sleep", MagicMock()),
            pytest.raises(RerankerAPIError, match="HTTP 503 too many times"),
        ):
            reranker._post_rerank({"model": "m", "query": "q", "documents": ["a"]})

        assert mock_client.post.call_count == 2


class TestRerankerClientLifecycle:
    """Lazy HTTP client, ``close``, and context manager."""

    def test_get_client_reuses_single_instance(self) -> None:
        mock_client = MagicMock()
        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=0,
        )
        with patch("cadence_md.app.reranker.httpx.Client", return_value=mock_client) as client_cls:
            assert reranker._get_client() is mock_client
            assert reranker._get_client() is mock_client
        client_cls.assert_called_once()

    def test_close_closes_underlying_client(self) -> None:
        mock_client = MagicMock()
        reranker = RerankerWrapper(
            model="m",
            top_k=5,
            return_score=True,
            base_url="http://h/v1",
            api_key="k",
            max_retries_on_rate_limit=0,
        )
        with patch("cadence_md.app.reranker.httpx.Client", return_value=mock_client):
            reranker._get_client()
            reranker.close()
        mock_client.close.assert_called_once()

    def test_context_manager_closes_client(self) -> None:
        mock_client = MagicMock()
        with (
            patch("cadence_md.app.reranker.httpx.Client", return_value=mock_client),
            RerankerWrapper(
                model="m",
                top_k=5,
                return_score=True,
                base_url="http://h/v1",
                api_key="k",
                max_retries_on_rate_limit=0,
            ) as reranker,
        ):
            reranker._get_client()
        mock_client.close.assert_called_once()


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

    def test_get_reranker_from_settings(self, settings) -> None:
        wrapper = get_reranker_from_settings(settings)

        assert isinstance(wrapper, RerankerWrapper)
        assert wrapper.model == settings.rag_config.reranker.model_name
        assert wrapper.top_k == settings.rag_config.reranker.top_k
        assert wrapper.return_score == settings.rag_config.reranker.return_score
