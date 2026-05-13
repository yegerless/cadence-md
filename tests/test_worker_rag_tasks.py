"""Unit tests for Celery RAG worker task flow."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Thread

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from cadence_md.db.base import Base
from cadence_md.db.enums import RAGRequestStatus
from cadence_md.db.models import RAGRequestLog
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
    """Minimal pipeline object for worker runtime tests."""

    def __init__(self) -> None:
        self.compiled = False

    def ensure_compiled(self) -> None:
        self.compiled = True


class DummyService:
    """Configurable RAG service stub."""

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


@pytest_asyncio.fixture
async def worker_session_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(rag_tasks, "AsyncSessionLocal", session_factory)
    set_rag_runtime_for_tests(None)

    yield session_factory

    shutdown_rag_runtime()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def worker_session_factory_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """On-disk SQLite so ``engine.dispose()`` between Celery-style runs keeps schema/data."""
    db_path = tmp_path / "rag_worker.sqlite"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(rag_tasks, "AsyncSessionLocal", session_factory)
    set_rag_runtime_for_tests(None)

    yield session_factory

    shutdown_rag_runtime()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def _response(
    answer: str = "Ответ [Doc 1].",
    *,
    flags: RAGFlags | None = None,
    clarification_question: str | None = None,
) -> RAGResponse:
    return RAGResponse(
        query="Вопрос",
        answer=answer,
        query_hash="hash",
        sources=[RAGSource(rank=1, doc_ref="[Doc 1]", filename="guideline.pdf", score=0.9)],
        latency=RAGLatency(
            input_guardrails=0.2,
            query_rewrite=0.5,
            qdrant=1.0,
            rerank=2.0,
            context_relevance=0.75,
            llm=3.0,
            answer_format=0.25,
            output_guardrails=0.4,
        ),
        flags=flags or RAGFlags(input_guardrail_passed=True, context_truncated=False),
        langfuse_trace_id="trace-1",
        clarification_question=clarification_question,
    )


async def _create_request(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    query: str = "Вопрос",
) -> uuid.UUID:
    async with session_factory() as session:
        user = await UserRepository(session).create_user(
            email=f"{uuid.uuid4()}@example.org",
            password_hash="hashed-password",
        )
        request_log = await RAGLogRepository(session).create_request(
            user_id=user.id,
            query=query,
            query_hash="hash",
        )
        await session.commit()
        return request_log.id


async def _get_status(
    session_factory: async_sessionmaker[AsyncSession],
    request_id: uuid.UUID,
) -> RAGRequestStatus:
    async with session_factory() as session:
        row = await RAGLogRepository(session).get_request(request_id)
        assert row is not None
        return row.status


async def _get_request_row(
    session_factory: async_sessionmaker[AsyncSession],
    request_id: uuid.UUID,
) -> RAGRequestLog:
    async with session_factory() as session:
        row = await RAGLogRepository(session).get_request(request_id)
        assert row is not None
        return row


@pytest.mark.asyncio
async def test_run_rag_request_happy_path(
    worker_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    request_id = await _create_request(worker_session_factory)
    service = DummyService(response=_response())
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-1")

    assert result == "succeeded"
    assert await _get_status(worker_session_factory, request_id) == RAGRequestStatus.SUCCEEDED
    assert len(service.calls) == 1
    async with worker_session_factory() as session:
        repo = RAGLogRepository(session)
        response = await repo.get_response_for_request(request_id)
        row = await repo.get_request(request_id)
        assert response is not None
        assert response.answer == "Ответ [Doc 1]."
        assert response.sources_json[0]["doc_ref"] == "[Doc 1]"
        assert response.latency_ms_json["input_guardrails"] == 0.2
        assert response.latency_ms_json["query_rewrite"] == 0.5
        assert response.latency_ms_json["context_relevance"] == 0.75
        assert response.latency_ms_json["answer_format"] == 0.25
        assert response.latency_ms_json["output_guardrails"] == 0.4
        assert response.flags_json["input_guardrail_passed"] is True
        assert response.langfuse_trace_id == "trace-1"
        assert row is not None
        assert row.celery_task_id == "task-1"


@pytest.mark.asyncio
async def test_run_rag_request_awaits_clarification_without_response_row(
    worker_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    request_id = await _create_request(worker_session_factory)
    service = DummyService(
        response=_response(
            "Уточните возраст пациента?",
            flags=RAGFlags(requires_clarification=True),
            clarification_question="Уточните возраст пациента?",
        )
    )
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-1")

    assert result == "awaiting_clarification"
    row = await _get_request_row(worker_session_factory, request_id)
    assert row.status == RAGRequestStatus.AWAITING_CLARIFICATION
    assert row.clarification_question == "Уточните возраст пациента?"
    assert row.clarification_requested_at is not None
    assert row.clarification_attempts == 1
    assert len(service.calls) == 1
    assert service.calls[0].allow_clarification is True
    async with worker_session_factory() as session:
        response = await RAGLogRepository(session).get_response_for_request(request_id)
        assert response is None


@pytest.mark.asyncio
async def test_run_rag_request_after_clarification_passes_answer(
    worker_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    request_id = await _create_request(worker_session_factory)
    async with worker_session_factory() as session:
        repo = RAGLogRepository(session)
        await repo.claim_queued_request(request_id, celery_task_id="task-1")
        await repo.mark_awaiting_clarification(request_id, question="Уточните возраст?")
        await repo.submit_clarification(
            request_id,
            answer="Пациент взрослый.",
            celery_task_id="task-2",
        )
        await session.commit()
    service = DummyService(response=_response("Финальный ответ [Doc 1]."))
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-2")

    assert result == "succeeded"
    assert await _get_status(worker_session_factory, request_id) == RAGRequestStatus.SUCCEEDED
    assert len(service.calls) == 1
    assert service.calls[0].clarification_answer == "Пациент взрослый."
    assert service.calls[0].allow_clarification is False
    async with worker_session_factory() as session:
        response = await RAGLogRepository(session).get_response_for_request(request_id)
        assert response is not None
        assert response.answer == "Финальный ответ [Doc 1]."


@pytest.mark.asyncio
async def test_run_rag_request_failure_path(
    worker_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    request_id = await _create_request(worker_session_factory)
    service = DummyService(error=RuntimeError("boom"))
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-1")

    assert result == "failed"
    assert await _get_status(worker_session_factory, request_id) == RAGRequestStatus.FAILED
    async with worker_session_factory() as session:
        response = await RAGLogRepository(session).get_response_for_request(request_id)
        assert response is not None
        assert response.error_type == "RuntimeError"
        assert response.error_message == "boom"


@pytest.mark.asyncio
async def test_run_rag_request_repeat_is_noop(
    worker_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    request_id = await _create_request(worker_session_factory)
    service = DummyService(response=_response())
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    first = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-1")
    second = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-1")

    assert first == "succeeded"
    assert second == "already_done"
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_celery_sync_entrypoint_two_fresh_loops_reuses_db(
    worker_session_factory_file: async_sessionmaker[AsyncSession],
) -> None:
    """Regression: each Celery task uses ``asyncio.run`` (new loop); pool must be reset."""
    request_id = await _create_request(worker_session_factory_file)
    service = DummyService(response=_response())
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    def run_twice() -> tuple[str, str]:
        first = rag_tasks._run_rag_request_in_fresh_event_loop(
            str(request_id), celery_task_id="task-loop-1"
        )
        second = rag_tasks._run_rag_request_in_fresh_event_loop(
            str(request_id), celery_task_id="task-loop-2"
        )
        return first, second

    first, second = await asyncio.to_thread(run_twice)

    assert first == "succeeded"
    assert second == "already_done"
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_run_rag_request_already_succeeded_is_noop(
    worker_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    request_id = await _create_request(worker_session_factory)
    async with worker_session_factory() as session:
        await RAGLogRepository(session).mark_succeeded(request_id)
        await session.commit()
    service = DummyService(error=AssertionError("should not run"))
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-1")

    assert result == "already_done"
    assert service.calls == []


@pytest.mark.asyncio
async def test_run_rag_request_cancel_before_start(
    worker_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    request_id = await _create_request(worker_session_factory)
    async with worker_session_factory() as session:
        await RAGLogRepository(session).request_cancel(request_id)
        await session.commit()
    service = DummyService(error=AssertionError("should not run"))
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-1")

    assert result == "cancelled"
    assert await _get_status(worker_session_factory, request_id) == RAGRequestStatus.CANCELLED
    assert service.calls == []


@pytest.mark.asyncio
async def test_run_rag_request_cancel_after_running_safe_point(
    worker_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    request_id = await _create_request(worker_session_factory)

    class CancellingService(DummyService):
        def run(self, request: RAGRequest) -> RAGResponse:
            self.calls.append(request)

            async def _cancel() -> None:
                async with worker_session_factory() as session:
                    await RAGLogRepository(session).request_cancel(request_id)
                    await session.commit()

            thread = Thread(target=lambda: asyncio.run(_cancel()))
            thread.start()
            thread.join()
            return _response()

    service = CancellingService(response=_response())
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(request_id), celery_task_id="task-1")

    assert result == "cancelled"
    assert await _get_status(worker_session_factory, request_id) == RAGRequestStatus.CANCELLED
    async with worker_session_factory() as session:
        assert await RAGLogRepository(session).get_response_for_request(request_id) is None


@pytest.mark.asyncio
async def test_retry_request_does_not_overwrite_original_response(
    worker_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    original_id = await _create_request(worker_session_factory, query="Оригинал")
    async with worker_session_factory() as session:
        repo = RAGLogRepository(session)
        await repo.create_response(rag_request_id=original_id, answer="Старый ответ.")
        await repo.mark_failed(original_id)
        retry_request = await repo.create_retry_request(original_request_id=original_id)
        retry_id = retry_request.id
        await session.commit()

    service = DummyService(response=_response("Новый ответ [Doc 1]."))
    set_rag_runtime_for_tests(RAGWorkerRuntime(service=service))

    result = await rag_tasks._run_rag_request_async(str(retry_id), celery_task_id="task-2")

    assert result == "succeeded"
    async with worker_session_factory() as session:
        repo = RAGLogRepository(session)
        original_response = await repo.get_response_for_request(original_id)
        retry_response = await repo.get_response_for_request(retry_id)
        assert original_response is not None
        assert original_response.answer == "Старый ответ."
        assert retry_response is not None
        assert retry_response.answer == "Новый ответ [Doc 1]."


def test_runtime_warmup_and_cleanup_hooks() -> None:
    closed: list[bool] = []
    service = DummyService(response=_response())
    runtime = RAGWorkerRuntime(service=service, close_hooks=[lambda: closed.append(True)])

    runtime.warmup()
    runtime.close()

    assert service.pipeline.compiled
    assert closed == [True]
