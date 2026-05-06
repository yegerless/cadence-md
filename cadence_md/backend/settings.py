"""Backend service settings for API, database, Redis, and Celery infrastructure."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Settings owned by the FastAPI backend and asynchronous service layer."""

    DATABASE_URL: str = "postgresql+asyncpg://cadence_md:cadence_md_dev@localhost:5432/cadence_md"
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    JWT_SECRET: str = Field(
        default="cadence-md-dev-only-secret-change-in-production!",
        min_length=32,
        description="HS256 signing secret for access tokens.",
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_SECONDS: int = Field(default=3600, ge=60, le=86400)

    AUTH_REGISTER_RATE_LIMIT_IP: str = Field(
        default="30/minute",
        description="SlowAPI / limits string for POST /auth/register per client IP.",
    )
    AUTH_LOGIN_RATE_LIMIT_IP: str = Field(
        default="30/minute",
        description="SlowAPI / limits string for POST /auth/login per client IP.",
    )
    CHAT_RATE_LIMIT_USER: str = Field(
        default="30/minute",
        description=(
            "SlowAPI limits string for POST /chat/messages and retry per authenticated user."
        ),
    )
    GLOBAL_RAG_QUEUE_MAX: int = Field(
        default=1000,
        ge=1,
        description="Max concurrent queued + running RAG requests across all users.",
    )

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        extra="ignore",
    )


backend_settings = BackendSettings()
