"""Async SQLAlchemy engine and session helpers for backend persistence."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine as _create_async_engine,
)

from cadence_md.backend.settings import backend_settings


def create_async_engine(database_url: str | None = None) -> AsyncEngine:
    """Create an async SQLAlchemy engine for the configured database URL."""
    return _create_async_engine(
        database_url or backend_settings.DATABASE_URL,
        pool_pre_ping=True,
    )


engine = create_async_engine()
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """Yield an async DB session for FastAPI dependencies or worker code."""
    async with AsyncSessionLocal() as session:
        yield session
