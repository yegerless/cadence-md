"""Unit tests for async database repositories."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from cadence_md.backend.schemas.chat import RAGRequestStatus as APIRAGRequestStatus
from cadence_md.db.base import Base
from cadence_md.db.enums import RAGRequestStatus
from cadence_md.db.models import RAGRequestLog, RAGResponseLog, User
from cadence_md.db.repositories import DuplicateRAGResponseError, RAGLogRepository, UserRepository


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Create an isolated async SQLite session with the ORM schema."""
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
    async with session_factory() as session:
        yield session

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_user_repository_creates_and_finds_user(db_session: AsyncSession) -> None:
    user_repository = UserRepository(db_session)

    user = await user_repository.create_user(
        email="doctor@example.org",
        password_hash="hashed-password",
        first_name="Ada",
        last_name="Lovelace",
    )

    assert isinstance(user, User)
    assert await user_repository.get_by_id(user.id) == user
    assert await user_repository.get_by_email("doctor@example.org") == user


@pytest.mark.asyncio
async def test_rag_request_status_values_match_api_enum() -> None:
    assert [status.value for status in RAGRequestStatus] == [
        status.value for status in APIRAGRequestStatus
    ]


@pytest.mark.asyncio
async def test_rag_log_repository_tracks_request_lifecycle(db_session: AsyncSession) -> None:
    user = await UserRepository(db_session).create_user(
        email="doctor@example.org",
        password_hash="hashed-password",
    )
    rag_repository = RAGLogRepository(db_session)

    request_log = await rag_repository.create_request(
        user_id=user.id,
        query="Как лечить артериальную гипертензию?",
        query_hash="query-sha256",
        conversation_id="conversation-1",
        idempotency_key="idempotency-1",
    )

    assert isinstance(request_log, RAGRequestLog)
    assert request_log.status == RAGRequestStatus.QUEUED
    assert request_log.retry_count == 0
    assert request_log.query_hash == "query-sha256"
    assert request_log.idempotency_key == "idempotency-1"

    running_request = await rag_repository.mark_running(
        request_log.id,
        celery_task_id="celery-task-1",
    )
    assert running_request is not None
    assert running_request.status == RAGRequestStatus.RUNNING
    assert running_request.celery_task_id == "celery-task-1"
    assert running_request.started_at is not None

    cancel_requested = await rag_repository.request_cancel(request_log.id)
    assert cancel_requested is not None
    assert cancel_requested.cancel_requested_at is not None

    failed_request = await rag_repository.mark_failed(request_log.id)
    assert failed_request is not None
    assert failed_request.status == RAGRequestStatus.FAILED
    assert failed_request.finished_at is not None

    retry_request = await rag_repository.create_retry_request(
        original_request_id=request_log.id,
        idempotency_key="retry-idempotency-1",
    )
    assert retry_request.original_request_id == request_log.id
    assert retry_request.retry_count == 1
    assert retry_request.status == RAGRequestStatus.QUEUED
    assert request_log.retry_count == 1


@pytest.mark.asyncio
async def test_rag_log_repository_marks_success_and_persists_response(
    db_session: AsyncSession,
) -> None:
    user = await UserRepository(db_session).create_user(
        email="doctor@example.org",
        password_hash="hashed-password",
    )
    rag_repository = RAGLogRepository(db_session)
    request_log = await rag_repository.create_request(
        user_id=user.id,
        query="Какие препараты первой линии?",
        query_hash="hash",
    )

    response = await rag_repository.create_response(
        rag_request_id=request_log.id,
        answer="Ответ с источником [Doc 1].",
        sources=[{"doc_ref": "[Doc 1]", "rank": 1}],
        latency_ms={"generate": 10.5},
        flags={"context_truncated": False},
        langfuse_trace_id="trace-1",
    )
    succeeded_request = await rag_repository.mark_succeeded(request_log.id)

    assert isinstance(response, RAGResponseLog)
    assert response.rag_request_id == request_log.id
    assert response.sources_json == [{"doc_ref": "[Doc 1]", "rank": 1}]
    assert response.latency_ms_json == {"generate": 10.5}
    assert response.flags_json == {"context_truncated": False}
    assert response.langfuse_trace_id == "trace-1"
    assert succeeded_request is not None
    assert succeeded_request.status == RAGRequestStatus.SUCCEEDED
    assert succeeded_request.finished_at is not None


@pytest.mark.asyncio
async def test_rag_log_repository_guards_against_duplicate_response(
    db_session: AsyncSession,
) -> None:
    user = await UserRepository(db_session).create_user(
        email="doctor@example.org",
        password_hash="hashed-password",
    )
    rag_repository = RAGLogRepository(db_session)
    request_log = await rag_repository.create_request(
        user_id=user.id,
        query="Нужна ли госпитализация?",
        query_hash="hash",
    )
    await rag_repository.create_response(
        rag_request_id=request_log.id,
        answer="Первый ответ.",
    )

    with pytest.raises(DuplicateRAGResponseError):
        await rag_repository.create_response(
            rag_request_id=request_log.id,
            answer="Повторный ответ.",
        )
