"""Shared observability settings for backend, worker, and RAG runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ObservabilitySettings(BaseSettings):
    """Configuration for logs, Prometheus metrics, and Langfuse tracing."""

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["plain", "json"] = "plain"
    LOG_REDACT_MEDICAL_QUERY: bool = True

    PROMETHEUS_ENABLED: bool = True
    WORKER_METRICS_PORT: int = Field(default=9100, ge=1, le=65535)
    PROMETHEUS_MULTIPROC_DIR: str | None = None

    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_BASE_URL: str | None = None
    LANGFUSE_HOST: str | None = None
    LANGFUSE_TRACE_QUERY_MODE: Literal["redacted", "hash", "full"] = "redacted"
    LANGFUSE_PROMPT_VERSION: str | None = None

    model_config = SettingsConfigDict(env_file=".env.dev", extra="ignore")


observability_settings = ObservabilitySettings()
