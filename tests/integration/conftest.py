"""Shared fixtures for integration tests with external Postgres and Redis."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cadence_md.backend.api.deps import get_backend_settings, get_rag_enqueue
from cadence_md.backend.limiter import limiter
from cadence_md.backend.main import create_app
from cadence_md.backend.services.rag_enqueue import NoopRAGEnqueueService
from cadence_md.backend.settings import BackendSettings
from cadence_md.db.base import Base
from cadence_md.db.session import get_async_session

INTEGRATION_DATABASE_URL_ENV = "INTEGRATION_DATABASE_URL"
INTEGRATION_REDIS_URL_ENV = "INTEGRATION_REDIS_URL"

DEFAULT_INTEGRATION_DATABASE_URL = (
    "postgresql+asyncpg://cadence_md:cadence_md_dev@127.0.0.1:55432/cadence_md_test"
)
DEFAULT_INTEGRATION_REDIS_URL = "redis://127.0.0.1:56379/0"


def get_integration_database_url() -> str:
    """Return DB URL from env with a safe local default."""
    return os.getenv(INTEGRATION_DATABASE_URL_ENV, DEFAULT_INTEGRATION_DATABASE_URL)


def get_integration_redis_url() -> str:
    """Return Redis URL from env with a safe local default."""
    return os.getenv(INTEGRATION_REDIS_URL_ENV, DEFAULT_INTEGRATION_REDIS_URL)


@pytest_asyncio.fixture(scope="session")
async def integration_database_url() -> str:
    """Provide integration DB URL and skip suite when Postgres is unavailable."""
    database_url = get_integration_database_url()
    probe_engine = create_async_engine(database_url, pool_pre_ping=True)
    try:
        async with probe_engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Integration Postgres is unavailable: {exc}")
    finally:
        await probe_engine.dispose()
    return database_url


@pytest_asyncio.fixture(scope="session")
async def integration_redis_url() -> str:
    """Provide integration Redis URL and skip suite when Redis is unavailable."""
    redis_url = get_integration_redis_url()
    client = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    try:
        await client.ping()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Integration Redis is unavailable: {exc}")
    finally:
        await client.aclose()
    return redis_url


@pytest_asyncio.fixture(scope="session")
async def integration_engine(integration_database_url: str):
    """Create shared async SQLAlchemy engine for integration tests."""
    engine = create_async_engine(integration_database_url, pool_pre_ping=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def integration_session_factory(integration_engine) -> async_sessionmaker[AsyncSession]:
    """Provide non-expiring async session factory bound to Postgres integration DB."""
    return async_sessionmaker(integration_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_integration_tables(
    integration_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    """Truncate mutable tables before each integration test for isolation."""
    async with integration_session_factory() as session:
        await session.execute(
            text("TRUNCATE TABLE rag_responses, rag_requests, users RESTART IDENTITY CASCADE")
        )
        await session.commit()
    yield


@pytest_asyncio.fixture
async def integration_settings(integration_redis_url: str) -> BackendSettings:
    """Settings tuned for deterministic integration test behavior."""
    return BackendSettings(
        JWT_SECRET="integration-test-jwt-secret-min-32-characters!",
        JWT_ACCESS_TOKEN_EXPIRE_SECONDS=3600,
        AUTH_REGISTER_RATE_LIMIT_IP="100/minute",
        AUTH_LOGIN_RATE_LIMIT_IP="100/minute",
        CHAT_RATE_LIMIT_USER="100/minute",
        GLOBAL_RAG_QUEUE_MAX=100,
        REDIS_URL=integration_redis_url,
        CELERY_BROKER_URL=integration_redis_url,
        CELERY_RESULT_BACKEND="redis://127.0.0.1:56379/1",
    )


@pytest_asyncio.fixture
async def integration_app(
    integration_session_factory: async_sessionmaker[AsyncSession],
    integration_settings: BackendSettings,
) -> AsyncIterator[FastAPI]:
    """Create ASGI app wired to integration Postgres and safe queue stub."""

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with integration_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app = create_app()
    app.dependency_overrides[get_async_session] = override_session
    app.dependency_overrides[get_backend_settings] = lambda: integration_settings
    app.dependency_overrides[get_rag_enqueue] = lambda: NoopRAGEnqueueService()
    app.state.settings = integration_settings

    yield app

    app.dependency_overrides.clear()
    limiter.reset()
