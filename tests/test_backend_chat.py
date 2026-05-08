"""Integration tests for backend chat (async RAG request) endpoints."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from cadence_md.backend.api.deps import get_backend_settings, get_health_service, get_rag_enqueue
from cadence_md.backend.limiter import limiter
from cadence_md.backend.main import create_app
from cadence_md.backend.schemas.health import HealthResponse, HealthStatus
from cadence_md.backend.schemas.limits import MAX_QUERY_LENGTH
from cadence_md.backend.services.rag_enqueue import (
    CeleryRAGEnqueueService,
    NoopRAGEnqueueService,
    make_rag_task_id,
)
from cadence_md.backend.settings import BackendSettings
from cadence_md.db.base import Base
from cadence_md.db.enums import RAGRequestStatus
from cadence_md.db.repositories.rag_logs import RAGLogRepository
from cadence_md.db.session import get_async_session

ChatBundle = tuple[FastAPI, BackendSettings, async_sessionmaker[AsyncSession]]
pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def chat_app_bundle() -> AsyncIterator[ChatBundle]:
    """ASGI app with SQLite DB, JWT settings, and session factory for direct repo calls."""
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

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    test_settings = BackendSettings(
        JWT_SECRET="unit-test-jwt-secret-min-32-characters!",
        JWT_ACCESS_TOKEN_EXPIRE_SECONDS=3600,
        AUTH_REGISTER_RATE_LIMIT_IP="100/minute",
        AUTH_LOGIN_RATE_LIMIT_IP="100/minute",
        CHAT_RATE_LIMIT_USER="100/minute",
        GLOBAL_RAG_QUEUE_MAX=100,
    )

    app = create_app()
    app.dependency_overrides[get_async_session] = override_session
    app.dependency_overrides[get_backend_settings] = lambda: test_settings
    app.dependency_overrides[get_rag_enqueue] = lambda: NoopRAGEnqueueService()
    app.state.settings = test_settings

    yield app, test_settings, session_factory

    app.dependency_overrides.clear()
    limiter.reset()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def _register(client: AsyncClient, email: str, password: str = "secure-password-here") -> str:
    reg = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    assert reg.status_code == 201, reg.text
    return reg.json()["access_token"]


@pytest.mark.asyncio
async def test_create_chat_message_returns_queued_and_enqueues(chat_app_bundle: ChatBundle) -> None:
    app, _settings, _sf = chat_app_bundle
    enqueue_calls: list[uuid.UUID] = []

    class RecordingEnqueue:
        async def enqueue(self, request_id: uuid.UUID) -> None:
            enqueue_calls.append(request_id)

    app.dependency_overrides[get_rag_enqueue] = lambda: RecordingEnqueue()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "doc@example.org")
        res = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Тестовый клинический вопрос?"},
        )
        assert res.status_code == 202
        body = res.json()
        assert body["status"] == "queued"
        rid = uuid.UUID(body["request_id"])
        assert enqueue_calls == [rid]

    async with _sf() as session:
        repo = RAGLogRepository(session)
        row = await repo.get_request(rid)
        assert row is not None
        assert row.celery_task_id == make_rag_task_id(rid)


@pytest.mark.asyncio
async def test_create_chat_message_still_queues_when_rag_health_unavailable(
    chat_app_bundle: ChatBundle,
) -> None:
    app, _settings, _sf = chat_app_bundle
    enqueue_calls: list[uuid.UUID] = []

    class RecordingEnqueue:
        async def enqueue(self, request_id: uuid.UUID) -> None:
            enqueue_calls.append(request_id)

    class UnreadyHealthService:
        async def rag(self) -> HealthResponse:
            return HealthResponse(
                status=HealthStatus.UNAVAILABLE,
                checks={"qdrant": HealthStatus.UNAVAILABLE},
            )

    app.dependency_overrides[get_rag_enqueue] = lambda: RecordingEnqueue()
    app.dependency_overrides[get_health_service] = lambda: UnreadyHealthService()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "rag-unready@example.org")
        health = await client.get("/api/v1/health/rag")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Можно ли поставить запрос в очередь при неготовом RAG?"},
        )

    assert health.status_code == 503
    assert created.status_code == 202
    assert created.json()["status"] == "queued"
    assert enqueue_calls


@pytest.mark.asyncio
async def test_get_own_message_status(chat_app_bundle: ChatBundle) -> None:
    app, _settings, _sf = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "poll@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Вопрос для polling"},
        )
        rid = created.json()["request_id"]
        poll = await client.get(
            f"/api/v1/chat/messages/{rid}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert poll.status_code == 200
        assert poll.json()["request_id"] == rid
        assert poll.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_get_other_users_message_returns_404(chat_app_bundle: ChatBundle) -> None:
    app, _settings, _sf = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token_a = await _register(client, "a@example.org")
        token_b = await _register(client, "b@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"query": "Секретный запрос"},
        )
        rid = created.json()["request_id"]
        leak = await client.get(
            f"/api/v1/chat/messages/{rid}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert leak.status_code == 404
        assert leak.json()["code"] == "not_found"


@pytest.mark.asyncio
async def test_get_awaiting_clarification_returns_question_and_no_cross_user_leak(
    chat_app_bundle: ChatBundle,
) -> None:
    app, _settings, session_factory = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token_a = await _register(client, "clarify-poll-a@example.org")
        token_b = await _register(client, "clarify-poll-b@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"query": "Вопрос требует уточнения"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.mark_running(rid, celery_task_id="task-1")
            await repo.mark_awaiting_clarification(
                rid,
                question="Уточните возраст пациента?",
            )
            await session.commit()

        own_poll = await client.get(
            f"/api/v1/chat/messages/{rid}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        leaked_poll = await client.get(
            f"/api/v1/chat/messages/{rid}",
            headers={"Authorization": f"Bearer {token_b}"},
        )

    assert own_poll.status_code == 200
    payload = own_poll.json()
    assert payload["status"] == "awaiting_clarification"
    assert payload["clarification"]["question"] == "Уточните возраст пациента?"
    assert payload["clarification"]["answered"] is False
    assert payload["clarification"]["requested_at"] is not None
    assert leaked_poll.status_code == 404


@pytest.mark.asyncio
async def test_submit_clarification_requeues_and_enqueues(chat_app_bundle: ChatBundle) -> None:
    app, _settings, session_factory = chat_app_bundle
    enqueue_calls: list[tuple[uuid.UUID, str | None]] = []

    class RecordingEnqueue:
        async def enqueue(self, request_id: uuid.UUID, *, task_id: str | None = None) -> str:
            enqueue_calls.append((request_id, task_id))
            return task_id or make_rag_task_id(request_id)

    app.dependency_overrides[get_rag_enqueue] = lambda: RecordingEnqueue()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "clarify-submit@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Нужны рекомендации"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.mark_running(rid, celery_task_id="task-1")
            await repo.mark_awaiting_clarification(rid, question="Уточните возраст?")
            await session.commit()

        submitted = await client.post(
            f"/api/v1/chat/messages/{rid}/clarification",
            headers={"Authorization": f"Bearer {token}"},
            json={"answer": "Пациент взрослый."},
        )

    assert submitted.status_code == 200
    assert submitted.json()["status"] == "queued"
    assert submitted.json()["clarification"] is None
    assert enqueue_calls[-1] == (rid, make_rag_task_id(rid, clarification_attempt=1))
    async with session_factory() as session:
        row = await RAGLogRepository(session).get_request(rid)
        assert row is not None
        assert row.status == RAGRequestStatus.QUEUED
        assert row.clarification_answer == "Пациент взрослый."
        assert row.celery_task_id == make_rag_task_id(rid, clarification_attempt=1)


@pytest.mark.asyncio
async def test_submit_clarification_wrong_status_returns_409(
    chat_app_bundle: ChatBundle,
) -> None:
    app, _settings, _sf = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "clarify409@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Обычный queued request"},
        )
        rid = created.json()["request_id"]
        submitted = await client.post(
            f"/api/v1/chat/messages/{rid}/clarification",
            headers={"Authorization": f"Bearer {token}"},
            json={"answer": "Ответ без запроса уточнения."},
        )

    assert submitted.status_code == 409
    assert submitted.json()["code"] == "clarification_not_allowed"


@pytest.mark.asyncio
async def test_submit_clarification_enqueue_error_keeps_answer(
    chat_app_bundle: ChatBundle,
) -> None:
    app, _settings, session_factory = chat_app_bundle

    class FailingEnqueue:
        async def enqueue(self, request_id: uuid.UUID, *, task_id: str | None = None) -> str:
            raise RuntimeError("broker down")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "clarify503@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Нужны уточнения"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.mark_running(rid, celery_task_id="task-1")
            await repo.mark_awaiting_clarification(rid, question="Уточните?")
            await session.commit()

        app.dependency_overrides[get_rag_enqueue] = lambda: FailingEnqueue()
        submitted = await client.post(
            f"/api/v1/chat/messages/{rid}/clarification",
            headers={"Authorization": f"Bearer {token}"},
            json={"answer": "Уточняющий ответ."},
        )

    assert submitted.status_code == 503
    assert submitted.json()["code"] == "rag_enqueue_failed"
    async with session_factory() as session:
        row = await RAGLogRepository(session).get_request(rid)
        assert row is not None
        assert row.status == RAGRequestStatus.QUEUED
        assert row.clarification_answer == "Уточняющий ответ."


@pytest.mark.asyncio
async def test_cancel_queued_marks_cancelled(chat_app_bundle: ChatBundle) -> None:
    app, _settings, _sf = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "cancelq@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Отмена в очереди"},
        )
        rid = created.json()["request_id"]
        cancel = await client.post(
            f"/api/v1/chat/messages/{rid}/cancel",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_running_sets_cancel_requested(chat_app_bundle: ChatBundle) -> None:
    app, _settings, session_factory = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "cancelr@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Отмена во время выполнения"},
        )
        rid = created.json()["request_id"]
        req_uuid = uuid.UUID(rid)

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.mark_running(req_uuid, celery_task_id="task-1")
            await session.commit()

        cancel = await client.post(
            f"/api/v1/chat/messages/{rid}/cancel",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert cancel.status_code == 200
        payload = cancel.json()
        assert payload["status"] == "running"

    async with session_factory() as session:
        repo = RAGLogRepository(session)
        row = await repo.get_request(req_uuid)
        assert row is not None
        assert row.cancel_requested_at is not None


@pytest.mark.asyncio
async def test_cancel_awaiting_clarification_marks_cancelled(
    chat_app_bundle: ChatBundle,
) -> None:
    app, _settings, session_factory = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "cancel-awaiting@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Отмена уточнения"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.mark_running(rid, celery_task_id="task-1")
            await repo.mark_awaiting_clarification(rid, question="Уточните?")
            await session.commit()

        cancel = await client.post(
            f"/api/v1/chat/messages/{rid}/cancel",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_terminal_returns_409(chat_app_bundle: ChatBundle) -> None:
    app, _settings, session_factory = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "cancel409@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Уже завершён"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.mark_failed(rid)
            await session.commit()

        cancel = await client.post(
            f"/api/v1/chat/messages/{rid}/cancel",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert cancel.status_code == 409
        assert cancel.json()["code"] == "cancellation_not_allowed"


@pytest.mark.asyncio
async def test_retry_after_failed_creates_new_request(chat_app_bundle: ChatBundle) -> None:
    app, _settings, session_factory = chat_app_bundle
    enqueue_calls: list[uuid.UUID] = []

    class RecordingEnqueue:
        async def enqueue(self, request_id: uuid.UUID) -> None:
            enqueue_calls.append(request_id)

    app.dependency_overrides[get_rag_enqueue] = lambda: RecordingEnqueue()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "retry@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Повтор после ошибки"},
        )
        orig_id = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.mark_failed(orig_id)
            await session.commit()

        retry = await client.post(
            f"/api/v1/chat/messages/{orig_id}/retry",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert retry.status_code == 202
        body = retry.json()
        new_id = uuid.UUID(body["request_id"])
        assert new_id != orig_id
        assert body["original_request_id"] == str(orig_id)
        assert body["status"] == "queued"
        assert enqueue_calls[-1] == new_id


@pytest.mark.asyncio
async def test_retry_while_queued_returns_409(chat_app_bundle: ChatBundle) -> None:
    app, _settings, _sf = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "retry409@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Ещё в очереди"},
        )
        rid = created.json()["request_id"]
        retry = await client.post(
            f"/api/v1/chat/messages/{rid}/retry",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert retry.status_code == 409
        assert retry.json()["code"] == "retry_not_allowed"


@pytest.mark.asyncio
async def test_retry_while_awaiting_clarification_returns_409(
    chat_app_bundle: ChatBundle,
) -> None:
    app, _settings, session_factory = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "retry-awaiting@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Retry уточнения"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.mark_running(rid, celery_task_id="task-1")
            await repo.mark_awaiting_clarification(rid, question="Уточните?")
            await session.commit()

        retry = await client.post(
            f"/api/v1/chat/messages/{rid}/retry",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert retry.status_code == 409
    assert retry.json()["code"] == "retry_not_allowed"


@pytest.mark.asyncio
async def test_idempotency_returns_same_request_without_double_enqueue(
    chat_app_bundle: ChatBundle,
) -> None:
    app, _settings, _sf = chat_app_bundle
    enqueue_calls: list[uuid.UUID] = []

    class RecordingEnqueue:
        async def enqueue(self, request_id: uuid.UUID) -> None:
            enqueue_calls.append(request_id)

    app.dependency_overrides[get_rag_enqueue] = lambda: RecordingEnqueue()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "idem@example.org")
        headers = {
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "idem-key-1",
        }
        body: dict[str, Any] = {"query": "Тот же запрос"}
        first = await client.post("/api/v1/chat/messages", headers=headers, json=body)
        second = await client.post("/api/v1/chat/messages", headers=headers, json=body)
        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["request_id"] == second.json()["request_id"]
        assert len(enqueue_calls) == 1


@pytest.mark.asyncio
async def test_idempotency_header_body_mismatch_returns_422(chat_app_bundle: ChatBundle) -> None:
    app, _settings, _sf = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "idem422@example.org")
        res = await client.post(
            "/api/v1/chat/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": "header-key",
            },
            json={"query": "Конфликт ключей", "idempotency_key": "body-key"},
        )
        assert res.status_code == 422
        assert res.json()["code"] == "idempotency_key_mismatch"


@pytest.mark.asyncio
async def test_global_queue_limit_returns_503(chat_app_bundle: ChatBundle) -> None:
    app, settings, _sf = chat_app_bundle
    tight = settings.model_copy(update={"GLOBAL_RAG_QUEUE_MAX": 1})
    app.dependency_overrides[get_backend_settings] = lambda: tight
    app.state.settings = tight

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token_a = await _register(client, "full-a@example.org")
        token_b = await _register(client, "full-b@example.org")
        first = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"query": "Занимаем слот"},
        )
        assert first.status_code == 202
        second = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"query": "Очередь полна"},
        )
        assert second.status_code == 503
        assert second.json()["code"] == "queue_limit_exceeded"


@pytest.mark.asyncio
async def test_chat_rate_limit_returns_429(chat_app_bundle: ChatBundle) -> None:
    app, settings, _sf = chat_app_bundle
    tight = settings.model_copy(update={"CHAT_RATE_LIMIT_USER": "2/minute"})
    app.dependency_overrides[get_backend_settings] = lambda: tight
    app.state.settings = tight

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "ratelimit@example.org")
        for _ in range(2):
            r = await client.post(
                "/api/v1/chat/messages",
                headers={"Authorization": f"Bearer {token}"},
                json={"query": "Нагрузка"},
            )
            assert r.status_code == 202
        blocked = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Слишком часто"},
        )
        assert blocked.status_code == 429
        assert blocked.json()["code"] == "rate_limit_exceeded"


@pytest.mark.asyncio
async def test_query_too_long_returns_422(chat_app_bundle: ChatBundle) -> None:
    app, _settings, _sf = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "longq@example.org")
        res = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "x" * (MAX_QUERY_LENGTH + 1)},
        )
        assert res.status_code == 422
        assert res.json()["code"] == "validation_error"


@pytest.mark.asyncio
async def test_get_succeeded_returns_answer(chat_app_bundle: ChatBundle) -> None:
    app, _settings, session_factory = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "ok@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Вопрос с ответом"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.create_response(
                rag_request_id=rid,
                answer="Краткий вывод: ответ.",
                sources=[
                    {
                        "rank": 1,
                        "doc_ref": "[Doc 1]",
                        "filename": "g.pdf",
                        "score": 0.9,
                    }
                ],
                latency_ms={
                    "query_rewrite": 2.0,
                    "context_relevance": 3.0,
                    "answer_format": 4.0,
                    "llm": 1.0,
                },
                flags={"context_truncated": False},
            )
            await repo.mark_succeeded(rid)
            await session.commit()

        poll = await client.get(
            f"/api/v1/chat/messages/{rid}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert poll.status_code == 200
        data = poll.json()
        assert data["status"] == "succeeded"
        assert data["answer"] is not None
        assert data["answer"]["answer"] == "Краткий вывод: ответ."
        assert len(data["answer"]["sources"]) == 1
        assert data["answer"]["sources"][0]["doc_ref"] == "[Doc 1]"
        assert data["answer"]["latency_ms"]["query_rewrite"] == 2.0
        assert data["answer"]["latency_ms"]["context_relevance"] == 3.0
        assert data["answer"]["latency_ms"]["answer_format"] == 4.0


@pytest.mark.asyncio
async def test_download_own_source_file(chat_app_bundle: ChatBundle, tmp_path: Path) -> None:
    app, settings, session_factory = chat_app_bundle
    corpus_dir = tmp_path / "corpus"
    source_dir = corpus_dir / "main_specialities"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "guideline.pdf"
    source_file.write_bytes(b"%PDF-1.4 test")
    docker_like_settings = settings.model_copy(update={"RAG_CORPUS_DIR": corpus_dir})
    app.dependency_overrides[get_backend_settings] = lambda: docker_like_settings
    app.state.settings = docker_like_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "download@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Вопрос с источником"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.create_response(
                rag_request_id=rid,
                answer="Ответ.",
                sources=[
                    {
                        "rank": 1,
                        "doc_ref": "[Doc 1]",
                        "filename": "guideline.pdf",
                        "source_path": "main_specialities/guideline.pdf",
                        "score": 0.9,
                    }
                ],
                latency_ms={},
                flags={},
            )
            await repo.mark_succeeded(rid)
            await session.commit()

        downloaded = await client.get(
            f"/api/v1/chat/messages/{rid}/sources/1/download",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert downloaded.status_code == 200
    assert downloaded.content == b"%PDF-1.4 test"
    assert "guideline.pdf" in downloaded.headers["content-disposition"]


@pytest.mark.asyncio
async def test_download_source_for_other_user_returns_404(
    chat_app_bundle: ChatBundle,
    tmp_path: Path,
) -> None:
    app, settings, session_factory = chat_app_bundle
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "main_specialities").mkdir(parents=True)
    (corpus_dir / "main_specialities" / "guideline.pdf").write_bytes(b"pdf")
    test_settings = settings.model_copy(update={"RAG_CORPUS_DIR": corpus_dir})
    app.dependency_overrides[get_backend_settings] = lambda: test_settings
    app.state.settings = test_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token_a = await _register(client, "download-a@example.org")
        token_b = await _register(client, "download-b@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"query": "Чужой источник"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.create_response(
                rag_request_id=rid,
                answer="Ответ.",
                sources=[
                    {
                        "rank": 1,
                        "doc_ref": "[Doc 1]",
                        "filename": "guideline.pdf",
                        "source_path": "main_specialities/guideline.pdf",
                    }
                ],
                latency_ms={},
                flags={},
            )
            await repo.mark_succeeded(rid)
            await session.commit()

        leaked = await client.get(
            f"/api/v1/chat/messages/{rid}/sources/1/download",
            headers={"Authorization": f"Bearer {token_b}"},
        )

    assert leaked.status_code == 404


@pytest.mark.asyncio
async def test_download_source_missing_rank_and_file_return_404(
    chat_app_bundle: ChatBundle,
    tmp_path: Path,
) -> None:
    app, settings, session_factory = chat_app_bundle
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    test_settings = settings.model_copy(update={"RAG_CORPUS_DIR": corpus_dir})
    app.dependency_overrides[get_backend_settings] = lambda: test_settings
    app.state.settings = test_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "download-missing@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Нет файла"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.create_response(
                rag_request_id=rid,
                answer="Ответ.",
                sources=[
                    {
                        "rank": 1,
                        "doc_ref": "[Doc 1]",
                        "filename": "missing.pdf",
                        "source_path": "main_specialities/missing.pdf",
                    }
                ],
                latency_ms={},
                flags={},
            )
            await repo.mark_succeeded(rid)
            await session.commit()

        missing_rank = await client.get(
            f"/api/v1/chat/messages/{rid}/sources/2/download",
            headers={"Authorization": f"Bearer {token}"},
        )
        missing_file = await client.get(
            f"/api/v1/chat/messages/{rid}/sources/1/download",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert missing_rank.status_code == 404
    assert missing_file.status_code == 404


@pytest.mark.asyncio
async def test_download_source_rejects_path_traversal(
    chat_app_bundle: ChatBundle,
    tmp_path: Path,
) -> None:
    app, settings, session_factory = chat_app_bundle
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (tmp_path / "secret.pdf").write_bytes(b"secret")
    test_settings = settings.model_copy(update={"RAG_CORPUS_DIR": corpus_dir})
    app.dependency_overrides[get_backend_settings] = lambda: test_settings
    app.state.settings = test_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "download-traversal@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Traversal"},
        )
        rid = uuid.UUID(created.json()["request_id"])

        async with session_factory() as session:
            repo = RAGLogRepository(session)
            await repo.create_response(
                rag_request_id=rid,
                answer="Ответ.",
                sources=[
                    {
                        "rank": 1,
                        "doc_ref": "[Doc 1]",
                        "filename": "secret.pdf",
                        "source_path": "../secret.pdf",
                    }
                ],
                latency_ms={},
                flags={},
            )
            await repo.mark_succeeded(rid)
            await session.commit()

        blocked = await client.get(
            f"/api/v1/chat/messages/{rid}/sources/1/download",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert blocked.status_code == 404


@pytest.mark.asyncio
async def test_cancel_and_retry_authorization_enforced(chat_app_bundle: ChatBundle) -> None:
    app, _settings, _sf = chat_app_bundle
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token_a = await _register(client, "ca@example.org")
        token_b = await _register(client, "cb@example.org")
        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"query": "Чужой cancel"},
        )
        rid = created.json()["request_id"]

        cancel_b = await client.post(
            f"/api/v1/chat/messages/{rid}/cancel",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert cancel_b.status_code == 404

        retry_b = await client.post(
            f"/api/v1/chat/messages/{rid}/retry",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert retry_b.status_code == 404


def test_chat_router_import_does_not_load_app_rag() -> None:
    """Subprocess isolation: chat router import must not load ``cadence_md.app.rag``."""
    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import cadence_md.backend.api.routers.chat\n"
        "sys.exit(1 if 'cadence_md.app.rag' in sys.modules else 0)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=env,
        check=False,
    )
    assert proc.returncode == 0


@pytest.mark.asyncio
async def test_celery_enqueue_service_sends_deterministic_task_id() -> None:
    sent: dict[str, Any] = {}

    class FakeCeleryApp:
        def send_task(self, name: str, **kwargs: Any) -> None:
            sent["name"] = name
            sent.update(kwargs)

    settings = BackendSettings(
        JWT_SECRET="unit-test-jwt-secret-min-32-characters!",
        CELERY_RAG_TASK_NAME="test.rag",
        CELERY_RAG_QUEUE_NAME="test-rag",
    )
    service = CeleryRAGEnqueueService(settings=settings, app=FakeCeleryApp())
    request_id = uuid.uuid4()

    task_id = await service.enqueue(request_id)

    assert task_id == make_rag_task_id(request_id)
    assert sent == {
        "name": "test.rag",
        "args": (str(request_id),),
        "task_id": task_id,
        "queue": "test-rag",
    }
