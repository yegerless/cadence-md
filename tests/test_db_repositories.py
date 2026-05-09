"""Unit tests for async database repositories."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from cadence_md.backend.schemas.chat import RAGRequestStatus as APIRAGRequestStatus
from cadence_md.db.base import Base
from cadence_md.db.enums import RAGRequestStatus
from cadence_md.db.models import ChatConversation, RAGRequestLog, RAGResponseLog, User
from cadence_md.db.repositories import (
    ChatConversationRepository,
    DuplicateRAGResponseError,
    RAGLogRepository,
    UserRepository,
)


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
async def test_chat_conversation_repository_lists_only_own_active_chats(
    db_session: AsyncSession,
) -> None:
    user_repository = UserRepository(db_session)
    user = await user_repository.create_user(
        email="chat-owner@example.org",
        password_hash="hashed-password",
    )
    other_user = await user_repository.create_user(
        email="other-chat-owner@example.org",
        password_hash="hashed-password",
    )
    chat_repository = ChatConversationRepository(db_session)
    older_chat = await chat_repository.create_chat(user_id=user.id, title="Older")
    newer_chat = await chat_repository.create_chat(user_id=user.id, title="Newer")
    other_chat = await chat_repository.create_chat(user_id=other_user.id, title="Other")

    await chat_repository.touch_chat_for_user(
        user_id=user.id,
        chat_id=older_chat.id,
        last_message_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    await chat_repository.touch_chat_for_user(
        user_id=user.id,
        chat_id=newer_chat.id,
        last_message_at=datetime(2026, 5, 2, tzinfo=UTC),
    )

    own_chats = await chat_repository.list_active_chats_for_user(user_id=user.id)
    assert [chat.id for chat in own_chats] == [newer_chat.id, older_chat.id]

    other_chats = await chat_repository.list_active_chats_for_user(user_id=other_user.id)
    assert [chat.id for chat in other_chats] == [other_chat.id]


@pytest.mark.asyncio
async def test_chat_conversation_repository_soft_delete_hides_chat_and_keeps_row(
    db_session: AsyncSession,
) -> None:
    user_repository = UserRepository(db_session)
    user = await user_repository.create_user(
        email="delete-chat-owner@example.org",
        password_hash="hashed-password",
    )
    other_user = await user_repository.create_user(
        email="delete-other-owner@example.org",
        password_hash="hashed-password",
    )
    chat_repository = ChatConversationRepository(db_session)
    chat = await chat_repository.create_chat(user_id=user.id, title="Delete me")
    other_chat = await chat_repository.create_chat(user_id=other_user.id, title="Not yours")

    assert (
        await chat_repository.get_active_chat_for_user(
            user_id=user.id,
            chat_id=other_chat.id,
        )
        is None
    )
    assert (
        await chat_repository.soft_delete_chat_for_user(
            user_id=user.id,
            chat_id=other_chat.id,
        )
        is None
    )

    deleted_chat = await chat_repository.soft_delete_chat_for_user(
        user_id=user.id,
        chat_id=chat.id,
    )

    assert deleted_chat is not None
    assert deleted_chat.deleted_at is not None
    assert await chat_repository.get_active_chat_for_user(user_id=user.id, chat_id=chat.id) is None
    assert await chat_repository.list_active_chats_for_user(user_id=user.id) == []
    assert await db_session.get(ChatConversation, chat.id) is not None


@pytest.mark.asyncio
async def test_chat_conversation_repository_lists_requests_and_keeps_messages_after_delete(
    db_session: AsyncSession,
) -> None:
    user_repository = UserRepository(db_session)
    user = await user_repository.create_user(
        email="chat-messages@example.org",
        password_hash="hashed-password",
    )
    other_user = await user_repository.create_user(
        email="chat-messages-other@example.org",
        password_hash="hashed-password",
    )
    chat_repository = ChatConversationRepository(db_session)
    rag_repository = RAGLogRepository(db_session)
    chat = await chat_repository.create_chat(user_id=user.id, title="Messages")
    other_chat = await chat_repository.create_chat(user_id=other_user.id, title="Other")

    second_request = await rag_repository.create_request(
        user_id=user.id,
        query="Second",
        query_hash="hash-2",
        chat_id=chat.id,
    )
    first_request = await rag_repository.create_request(
        user_id=user.id,
        query="First",
        query_hash="hash-1",
        chat_id=chat.id,
    )
    await rag_repository.create_request(
        user_id=user.id,
        query="Unrelated",
        query_hash="hash-unrelated",
    )
    await rag_repository.create_request(
        user_id=other_user.id,
        query="Other user",
        query_hash="hash-other",
        chat_id=other_chat.id,
    )
    second_request.created_at = datetime(2026, 5, 2, tzinfo=UTC)
    first_request.created_at = datetime(2026, 5, 1, tzinfo=UTC)
    await db_session.flush()

    requests = await chat_repository.list_requests_by_chat_for_user(
        user_id=user.id,
        chat_id=chat.id,
    )
    assert [request.id for request in requests] == [first_request.id, second_request.id]

    await chat_repository.soft_delete_chat_for_user(user_id=user.id, chat_id=chat.id)

    assert await db_session.get(RAGRequestLog, first_request.id) is not None
    assert await db_session.get(RAGRequestLog, second_request.id) is not None


@pytest.mark.asyncio
async def test_rag_log_repository_binds_request_to_chat_and_retry_inherits_chat_id(
    db_session: AsyncSession,
) -> None:
    user = await UserRepository(db_session).create_user(
        email="retry-chat@example.org",
        password_hash="hashed-password",
    )
    chat_repository = ChatConversationRepository(db_session)
    chat = await chat_repository.create_chat(user_id=user.id, title="Retry chat")
    rag_repository = RAGLogRepository(db_session)

    request_log = await rag_repository.create_request(
        user_id=user.id,
        query="Какие препараты первой линии?",
        query_hash="hash",
        chat_id=chat.id,
    )
    retry_request = await rag_repository.create_retry_request(
        original_request_id=request_log.id,
        idempotency_key="retry-chat-idempotency",
    )

    assert request_log.chat_id == chat.id
    assert retry_request.chat_id == chat.id
    assert retry_request.original_request_id == request_log.id


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
async def test_rag_log_repository_counts_active_requests(db_session: AsyncSession) -> None:
    user = await UserRepository(db_session).create_user(
        email="count@example.org",
        password_hash="hashed-password",
    )
    repo = RAGLogRepository(db_session)
    r1 = await repo.create_request(
        user_id=user.id,
        query="Q1",
        query_hash="h1",
    )
    r2 = await repo.create_request(
        user_id=user.id,
        query="Q2",
        query_hash="h2",
    )
    assert await repo.count_active_requests() == 2
    await repo.mark_running(r1.id)
    assert await repo.count_active_requests() == 2
    await repo.mark_succeeded(r2.id)
    assert await repo.count_active_requests() == 1
    await repo.mark_awaiting_clarification(r1.id, question="Уточните возраст пациента?")
    assert await repo.count_active_requests() == 0


@pytest.mark.asyncio
async def test_rag_log_repository_find_by_idempotency_key(db_session: AsyncSession) -> None:
    user = await UserRepository(db_session).create_user(
        email="idem-repo@example.org",
        password_hash="hashed-password",
    )
    repo = RAGLogRepository(db_session)
    created = await repo.create_request(
        user_id=user.id,
        query="Same",
        query_hash="hx",
        idempotency_key="shared-key",
    )
    found = await repo.get_request_by_idempotency_key(
        user_id=user.id,
        idempotency_key="shared-key",
    )
    assert found is not None
    assert found.id == created.id


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


@pytest.mark.asyncio
async def test_rag_log_repository_guarded_status_transitions(db_session: AsyncSession) -> None:
    user = await UserRepository(db_session).create_user(
        email="guarded@example.org",
        password_hash="hashed-password",
    )
    repo = RAGLogRepository(db_session)
    request_log = await repo.create_request(
        user_id=user.id,
        query="Guarded transition",
        query_hash="hash",
    )

    claimed = await repo.claim_queued_request(request_log.id, celery_task_id="task-1")
    assert claimed is not None
    assert claimed.status == RAGRequestStatus.RUNNING
    assert claimed.celery_task_id == "task-1"
    assert claimed.started_at is not None

    second_claim = await repo.claim_queued_request(request_log.id, celery_task_id="task-2")
    assert second_claim is None

    succeeded = await repo.mark_succeeded_from_running(request_log.id)
    assert succeeded is not None
    assert succeeded.status == RAGRequestStatus.SUCCEEDED

    failed_after_terminal = await repo.mark_failed_from_running(request_log.id)
    assert failed_after_terminal is None


@pytest.mark.asyncio
async def test_rag_log_repository_clarification_guarded_transitions(
    db_session: AsyncSession,
) -> None:
    user = await UserRepository(db_session).create_user(
        email="clarify@example.org",
        password_hash="hashed-password",
    )
    repo = RAGLogRepository(db_session)
    request_log = await repo.create_request(
        user_id=user.id,
        query="Нужны рекомендации?",
        query_hash="hash",
    )

    not_running = await repo.mark_awaiting_clarification(
        request_log.id,
        question="Уточните?",
    )
    assert not_running is None

    await repo.claim_queued_request(request_log.id, celery_task_id="task-1")
    awaiting = await repo.mark_awaiting_clarification(
        request_log.id,
        question="Уточните возраст пациента?",
    )
    assert awaiting is not None
    assert awaiting.status == RAGRequestStatus.AWAITING_CLARIFICATION
    assert awaiting.clarification_question == "Уточните возраст пациента?"
    assert awaiting.clarification_requested_at is not None
    assert awaiting.clarification_attempts == 1
    assert awaiting.finished_at is None

    second_awaiting = await repo.mark_awaiting_clarification(
        request_log.id,
        question="Повторный вопрос?",
    )
    assert second_awaiting is None

    submitted = await repo.submit_clarification(
        request_log.id,
        answer="Пациент взрослый.",
        celery_task_id="task-2",
    )
    assert submitted is not None
    assert submitted.status == RAGRequestStatus.QUEUED
    assert submitted.clarification_answer == "Пациент взрослый."
    assert submitted.clarification_answered_at is not None
    assert submitted.celery_task_id == "task-2"

    duplicate_submit = await repo.submit_clarification(
        request_log.id,
        answer="Повторный ответ.",
        celery_task_id="task-3",
    )
    assert duplicate_submit is None


@pytest.mark.asyncio
async def test_rag_log_repository_queued_task_id_guard(db_session: AsyncSession) -> None:
    user = await UserRepository(db_session).create_user(
        email="taskid@example.org",
        password_hash="hashed-password",
    )
    repo = RAGLogRepository(db_session)
    request_log = await repo.create_request(
        user_id=user.id,
        query="Task id",
        query_hash="hash",
    )

    updated = await repo.set_queued_task_id(request_log.id, celery_task_id="task-1")
    assert updated is not None
    assert updated.celery_task_id == "task-1"

    await repo.claim_queued_request(request_log.id, celery_task_id="task-1")
    stale_update = await repo.set_queued_task_id(request_log.id, celery_task_id="task-2")
    assert stale_update is None


@pytest.mark.asyncio
async def test_rag_log_repository_has_response_for_request(db_session: AsyncSession) -> None:
    user = await UserRepository(db_session).create_user(
        email="has-response@example.org",
        password_hash="hashed-password",
    )
    repo = RAGLogRepository(db_session)
    request_log = await repo.create_request(
        user_id=user.id,
        query="Has response",
        query_hash="hash",
    )

    assert not await repo.has_response_for_request(request_log.id)
    await repo.create_response(rag_request_id=request_log.id, answer="Ответ.")
    assert await repo.has_response_for_request(request_log.id)
