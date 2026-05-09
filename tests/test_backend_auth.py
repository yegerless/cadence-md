"""Integration tests for backend authentication and profile endpoints."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from cadence_md.backend.api.deps import get_backend_settings
from cadence_md.backend.limiter import limiter
from cadence_md.backend.main import create_app
from cadence_md.backend.settings import BackendSettings
from cadence_md.db.base import Base
from cadence_md.db.session import get_async_session

AuthAppPair = tuple[FastAPI, BackendSettings]
pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def auth_app_and_settings() -> AsyncIterator[AuthAppPair]:
    """ASGI app with SQLite DB and JWT settings suitable for tests."""
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
    )

    app = create_app()
    app.dependency_overrides[get_async_session] = override_session
    app.dependency_overrides[get_backend_settings] = lambda: test_settings
    app.state.settings = test_settings

    yield app, test_settings

    app.dependency_overrides.clear()
    limiter.reset()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_register_login_and_profile(auth_app_and_settings: AuthAppPair) -> None:
    app, _settings = auth_app_and_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "Doctor@Example.org",
                "password": "secure-password-here",
                "first_name": "Ada",
                "last_name": "Lovelace",
            },
        )
        assert reg.status_code == 201
        token_payload = reg.json()
        assert token_payload["token_type"] == "bearer"
        assert "access_token" in token_payload
        assert token_payload["expires_in"] == 3600

        login = await client.post(
            "/api/v1/auth/login",
            json={"email": "doctor@example.org", "password": "secure-password-here"},
        )
        assert login.status_code == 200
        access_token = login.json()["access_token"]

        me = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert me.status_code == 200
        profile = me.json()
        assert profile["email"] == "doctor@example.org"
        assert profile["first_name"] == "Ada"
        assert profile["last_name"] == "Lovelace"
        assert "id" in profile


@pytest.mark.asyncio
async def test_register_duplicate_email_returns_409(auth_app_and_settings: AuthAppPair) -> None:
    app, _settings = auth_app_and_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "email": "dup@example.org",
            "password": "secure-password-here",
        }
        first = await client.post("/api/v1/auth/register", json=body)
        assert first.status_code == 201
        second = await client.post("/api/v1/auth/register", json=body)
        assert second.status_code == 409
        err = second.json()
        assert err["code"] == "email_already_registered"


@pytest.mark.asyncio
async def test_login_wrong_password_returns_401(auth_app_and_settings: AuthAppPair) -> None:
    app, _settings = auth_app_and_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/api/v1/auth/register",
            json={"email": "user@example.org", "password": "correct-password"},
        )
        bad = await client.post(
            "/api/v1/auth/login",
            json={"email": "user@example.org", "password": "wrong-password"},
        )
        assert bad.status_code == 401
        assert bad.json()["code"] == "invalid_credentials"


@pytest.mark.asyncio
async def test_profile_without_token_returns_401(auth_app_and_settings: AuthAppPair) -> None:
    app, _settings = auth_app_and_settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        me = await client.get("/api/v1/users/me")
        assert me.status_code == 401
        assert me.json()["code"] == "not_authenticated"


@pytest.mark.asyncio
async def test_register_rate_limit_returns_429(auth_app_and_settings: AuthAppPair) -> None:
    app, settings = auth_app_and_settings
    tight = settings.model_copy(update={"AUTH_REGISTER_RATE_LIMIT_IP": "2/minute"})
    app.state.settings = tight

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(2):
            r = await client.post(
                "/api/v1/auth/register",
                json={
                    "email": f"u{i}@example.org",
                    "password": "secure-password-here",
                },
            )
            assert r.status_code == 201, r.text

        blocked = await client.post(
            "/api/v1/auth/register",
            json={"email": "u3@example.org", "password": "secure-password-here"},
        )
        assert blocked.status_code == 429
        assert blocked.json()["code"] == "rate_limit_exceeded"


def test_backend_import_does_not_load_app_rag() -> None:
    """Subprocess isolation: importing backend main must not load ``cadence_md.app.rag``."""
    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import cadence_md.backend.main\n"
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
