"""Backend service settings for API, database, Redis, and Celery infrastructure."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Settings owned by the FastAPI backend and asynchronous service layer."""

    DATABASE_URL: str = "postgresql+asyncpg://cadence_md:cadence_md_dev@localhost:5432/cadence_md"
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        extra="ignore",
    )


backend_settings = BackendSettings()
