"""Backend integration tests against external Postgres and Redis."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cadence_md.backend.api.deps import get_backend_settings
from cadence_md.backend.schemas.limits import MAX_QUERY_LENGTH
from cadence_md.backend.settings import BackendSettings
from cadence_md.db.repositories.rag_logs import RAGLogRepository

pytestmark = pytest.mark.integration


async def _register(client: AsyncClient, email: str, password: str = "secure-password-here") -> str:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
    )
    assert response.status_code == 201, response.text
    return response.json()["access_token"]


@pytest.mark.asyncio
async def test_register_login_profile_and_redis_available(
    integration_app: FastAPI,
    integration_redis_url: str,
) -> None:
    transport = ASGITransport(app=integration_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        register = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "doctor@example.org",
                "password": "secure-password-here",
                "first_name": "Ada",
                "last_name": "Lovelace",
            },
        )
        assert register.status_code == 201

        login = await client.post(
            "/api/v1/auth/login",
            json={"email": "doctor@example.org", "password": "secure-password-here"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]

        profile = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert profile.status_code == 200
        assert profile.json()["email"] == "doctor@example.org"

    assert integration_redis_url.startswith("redis://")


@pytest.mark.asyncio
async def test_create_poll_cancel_retry_and_authorization(
    integration_app: FastAPI,
    integration_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    transport = ASGITransport(app=integration_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token_a = await _register(client, "a@example.org")
        token_b = await _register(client, "b@example.org")

        created = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"query": "Тестовый клинический вопрос"},
        )
        assert created.status_code == 202
        request_id = created.json()["request_id"]

        poll = await client.get(
            f"/api/v1/chat/messages/{request_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert poll.status_code == 200
        assert poll.json()["status"] == "queued"

        leak_poll = await client.get(
            f"/api/v1/chat/messages/{request_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert leak_poll.status_code == 404

        cancel = await client.post(
            f"/api/v1/chat/messages/{request_id}/cancel",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelled"

        leak_cancel = await client.post(
            f"/api/v1/chat/messages/{request_id}/cancel",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert leak_cancel.status_code == 404

        retry = await client.post(
            f"/api/v1/chat/messages/{request_id}/retry",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert retry.status_code == 202
        retry_id = retry.json()["request_id"]
        assert retry_id != request_id

        leak_retry = await client.post(
            f"/api/v1/chat/messages/{request_id}/retry",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert leak_retry.status_code == 404

    async with integration_session_factory() as session:
        repo = RAGLogRepository(session)
        retry_row = await repo.get_request(uuid.UUID(retry_id))
        assert retry_row is not None
        assert retry_row.original_request_id == uuid.UUID(request_id)


@pytest.mark.asyncio
async def test_idempotency_key_returns_same_request_id(integration_app: FastAPI) -> None:
    transport = ASGITransport(app=integration_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "idem@example.org")
        headers = {
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "idem-1",
        }
        payload: dict[str, Any] = {"query": "Одинаковый запрос"}
        first = await client.post("/api/v1/chat/messages", headers=headers, json=payload)
        second = await client.post("/api/v1/chat/messages", headers=headers, json=payload)
        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["request_id"] == second.json()["request_id"]


@pytest.mark.asyncio
async def test_rate_limits_for_auth_and_chat(
    integration_app: FastAPI,
    integration_settings: BackendSettings,
) -> None:
    tight = integration_settings.model_copy(
        update={
            "AUTH_REGISTER_RATE_LIMIT_IP": "1/minute",
            "AUTH_LOGIN_RATE_LIMIT_IP": "1/minute",
            "CHAT_RATE_LIMIT_USER": "1/minute",
        }
    )
    integration_app.dependency_overrides[get_backend_settings] = lambda: tight
    integration_app.state.settings = tight

    transport = ASGITransport(app=integration_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first_register = await client.post(
            "/api/v1/auth/register",
            json={"email": "u1@example.org", "password": "secure-password-here"},
        )
        assert first_register.status_code == 201

        blocked_register = await client.post(
            "/api/v1/auth/register",
            json={"email": "u2@example.org", "password": "secure-password-here"},
        )
        assert blocked_register.status_code == 429

        first_login = await client.post(
            "/api/v1/auth/login",
            json={"email": "u1@example.org", "password": "secure-password-here"},
        )
        assert first_login.status_code == 200
        blocked_login = await client.post(
            "/api/v1/auth/login",
            json={"email": "u1@example.org", "password": "secure-password-here"},
        )
        assert blocked_login.status_code == 429

        token = first_login.json()["access_token"]
        first_chat = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Первый запрос"},
        )
        assert first_chat.status_code == 202
        blocked_chat = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "Второй запрос"},
        )
        assert blocked_chat.status_code == 429


@pytest.mark.asyncio
async def test_max_query_length_validation(integration_app: FastAPI) -> None:
    transport = ASGITransport(app=integration_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        token = await _register(client, "long@example.org")
        response = await client.post(
            "/api/v1/chat/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "x" * (MAX_QUERY_LENGTH + 1)},
        )
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"
