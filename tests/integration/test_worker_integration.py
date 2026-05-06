"""Worker integration tests on Postgres with mocked RAG runtime."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cadence_md.db.enums import RAGRequestStatus
from cadence_md.db.repositories import RAGLogRepository, UserRepository
from cadence_md.rag.contracts import RAGFlags, RAGLatency, RAGRequest, RAGResponse, RAGSource
from cadence_md.workers import rag_tasks
from cadence_md.workers.rag_bootstrap import (
    RAGWorkerRuntime,
    set_rag_runtime_for_tests,
    shutdown_rag_runtime,
)

pytestmark = pytest.mark.integration


class DummyPipeline:
    """Minimal pipeline stub for worker warmup."""

    def ensure_compiled(self) -> None:
        return None


class DummyService:
    """Controllable fake RAG service for worker task integration tests."""

    def __init__(self, response: RAGResponse | None = None, error: Exception | None = None) -> None:
        self.pipeline = DummyPipeline()
        self.response = response
        self.error = error
        self.calls: list[RAGRequest] = []

    def run(self, request: RAGRequest) -> RAGResponse:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


@pytest.fixture(autouse=True)
def _worker_runtime_cleanup() -> None:
    set_rag_runtime_for_tests(None)
    yield
    shutdown_rag_runtime()


def _response(answer: str) -> RAGResponse:
    return RAGResponse(
        query="Вопрос",
        answer=answer,
        query_hash="hash",
        sources=[RAGSource(rank=1, doc_ref="[Doc 1]", filename="g.pdf", score=0.91)],
        latency=RAGLatency(qdrant=1.0, rerank=1.0, llm=1.0),
        flags=RAGFlags(context_truncated=False),
    )


async def _create_request(session_factory: async_sessionmaker[AsyncSession]) -> uuid.UUID:
    async with session_factory() as session:
        user = await UserRepository(session).create_user(
            email=f"{uuid.uuid4()}@example.org",
            password_hash="hash",
        )
        row = await RAGLogRepository(session).create_request(
            user_id=user.id,
            query="Вопрос",
            query_hash="hash",
        )
        await session.commit()
        return row.id


@pytest.mark.asyncio
async def test_worker_happy_path_creates_response(
    integration_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_tasks, "AsyncSessionLocal", integration_session_factory)
    request_id = await _create_request(integration_session_factory)
    service = DummyService(response=_response("Успешный ответ"))
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-happy")

    assert result == "succeeded"
    async with integration_session_factory() as session:
        repo = RAGLogRepository(session)
        request_row = await repo.get_request(request_id)
        response_row = await repo.get_response_for_request(request_id)
        assert request_row is not None
        assert request_row.status == RAGRequestStatus.SUCCEEDED
        assert response_row is not None
        assert response_row.answer == "Успешный ответ"


@pytest.mark.asyncio
async def test_worker_failure_path_marks_failed(
    integration_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_tasks, "AsyncSessionLocal", integration_session_factory)
    request_id = await _create_request(integration_session_factory)
    service = DummyService(error=RuntimeError("boom"))
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-fail")

    assert result == "failed"
    async with integration_session_factory() as session:
        repo = RAGLogRepository(session)
        request_row = await repo.get_request(request_id)
        response_row = await repo.get_response_for_request(request_id)
        assert request_row is not None
        assert request_row.status == RAGRequestStatus.FAILED
        assert response_row is not None
        assert response_row.error_type == "RuntimeError"
        assert response_row.error_message == "boom"


@pytest.mark.asyncio
async def test_worker_repeated_run_does_not_create_second_response(
    integration_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_tasks, "AsyncSessionLocal", integration_session_factory)
    request_id = await _create_request(integration_session_factory)
    service = DummyService(response=_response("Однократный ответ"))
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    first = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-repeat")
    second = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-repeat")

    assert first == "succeeded"
    assert second == "already_done"
    async with integration_session_factory() as session:
        repo = RAGLogRepository(session)
        response_row = await repo.get_response_for_request(request_id)
        assert response_row is not None
        assert response_row.answer == "Однократный ответ"
    assert len(service.calls) == 1
