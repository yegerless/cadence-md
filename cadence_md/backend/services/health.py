"""Dependency health checks for backend and RAG readiness."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from types import SimpleNamespace
from typing import Any

import httpx
from redis import asyncio as redis_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from cadence_md.backend.schemas.health import HealthCheckDetail, HealthResponse, HealthStatus
from cadence_md.backend.settings import BackendSettings

logger = logging.getLogger(__name__)

CheckCallable = Callable[[], Awaitable[HealthCheckDetail]]


class HealthService:
    """Run lightweight dependency checks without constructing the full RAG stack."""

    def __init__(self, backend_settings: BackendSettings, app_settings: Any) -> None:
        self._backend_settings = backend_settings
        self._app_settings = app_settings
        self._timeout_seconds = backend_settings.HEALTH_CHECK_TIMEOUT_SECONDS

    async def live(self) -> HealthResponse:
        """Return process liveness without external checks."""
        return self._response({"backend": self._ok("Backend process is alive.")})

    async def ready(self) -> HealthResponse:
        """Return readiness for backend traffic and RAG dependencies."""
        checks = await self._run_checks(
            {
                "backend": lambda: self._instant_ok("Backend process is alive."),
                "postgres": self.check_postgres,
                "redis": lambda: self.check_redis(
                    name="redis",
                    url=self._backend_settings.REDIS_URL,
                ),
                "qdrant": self.check_qdrant_availability,
                "qdrant_schema": self.check_qdrant_schema,
                "inference": self.check_inference_models,
            }
        )
        return self._response(checks)

    async def rag(self) -> HealthResponse:
        """Return detailed readiness for RAG dependencies and queue broker."""
        checks = await self._run_checks(
            {
                "qdrant": self.check_qdrant_availability,
                "qdrant_schema": self.check_qdrant_schema,
                "inference": self.check_inference_models,
                "queue": lambda: self.check_redis(
                    name="queue",
                    url=self._backend_settings.CELERY_BROKER_URL,
                ),
            }
        )
        return self._response(checks)

    async def check_postgres(self) -> HealthCheckDetail:
        """Check Postgres connectivity with a single ``SELECT 1``."""
        started = time.perf_counter()
        engine = create_async_engine(self._backend_settings.DATABASE_URL, pool_pre_ping=True)
        try:
            async with engine.connect() as connection:
                await asyncio.wait_for(
                    connection.execute(text("SELECT 1")),
                    timeout=self._timeout_seconds,
                )
        except Exception as exc:
            self._log_unavailable("postgres", exc)
            return self._unavailable(
                "Postgres connectivity check failed.",
                started=started,
                error_type=type(exc).__name__,
            )
        finally:
            await engine.dispose()
        return self._ok("Postgres is reachable.", started=started)

    async def check_redis(self, *, name: str, url: str) -> HealthCheckDetail:
        """Check Redis connectivity using ``PING``."""
        started = time.perf_counter()
        client = redis_asyncio.from_url(
            url,
            socket_connect_timeout=self._timeout_seconds,
            socket_timeout=self._timeout_seconds,
        )
        try:
            pong = await asyncio.wait_for(client.ping(), timeout=self._timeout_seconds)
            if pong is not True:
                return self._unavailable(
                    f"{name.capitalize()} ping returned an unexpected response.",
                    started=started,
                )
        except Exception as exc:
            self._log_unavailable(name, exc)
            return self._unavailable(
                f"{name.capitalize()} connectivity check failed.",
                started=started,
                error_type=type(exc).__name__,
            )
        finally:
            await client.aclose()
        return self._ok(f"{name.capitalize()} is reachable.", started=started)

    async def check_qdrant_availability(self) -> HealthCheckDetail:
        """Check that Qdrant answers a lightweight collections request."""
        started = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._list_qdrant_collections),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            self._log_unavailable("qdrant", exc)
            return self._unavailable(
                "Qdrant availability check failed.",
                started=started,
                error_type=type(exc).__name__,
            )
        return self._ok("Qdrant is reachable.", started=started)

    async def check_qdrant_schema(self) -> HealthCheckDetail:
        """Validate Qdrant collection schema against the current RAG configuration."""
        started = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._ensure_qdrant_schema),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            self._log_unavailable("qdrant_schema", exc)
            return self._unavailable(
                "Qdrant collection schema check failed.",
                started=started,
                error_type=type(exc).__name__,
            )

        collection_name = self._app_settings.rag_config.qdrant_config.collection_name
        return self._ok(
            "Qdrant collection schema matches the RAG configuration.",
            started=started,
            details={"collection": collection_name},
        )

    async def check_inference_models(self) -> HealthCheckDetail:
        """Check the OpenAI-compatible ``/models`` endpoint without running inference."""
        started = time.perf_counter()
        models_url = self._models_url()
        headers = self._inference_headers()
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.get(models_url, headers=headers)
        except Exception as exc:
            self._log_unavailable("inference", exc)
            return self._unavailable(
                "Inference API is unreachable.",
                started=started,
                error_type=type(exc).__name__,
            )

        return self._inference_response_detail(response, started=started)

    def _inference_response_detail(
        self,
        response: httpx.Response,
        *,
        started: float,
    ) -> HealthCheckDetail:
        if response.status_code in {404, 405, 501}:
            return self._degraded(
                "Inference API is reachable, but /models is not supported.",
                started=started,
                details={"status_code": response.status_code},
            )
        if response.status_code >= 400:
            return self._unavailable(
                "Inference API /models check failed.",
                started=started,
                details={"status_code": response.status_code},
            )

        try:
            payload = response.json()
        except ValueError:
            return self._degraded(
                "Inference API /models response is not JSON.",
                started=started,
            )

        model_ids = self._extract_model_ids(payload)
        if not model_ids:
            return self._degraded(
                "Inference API is reachable, but model ids could not be verified.",
                started=started,
            )

        expected = self._expected_model_names()
        missing = sorted(expected - model_ids)
        if missing:
            return self._unavailable(
                "Inference API is missing required RAG models.",
                started=started,
                details={"missing_models": missing},
            )

        return self._ok(
            "Inference API exposes required RAG models.",
            started=started,
            details={"models_checked": sorted(expected)},
        )

    async def _run_checks(
        self,
        check_factories: Mapping[str, CheckCallable],
    ) -> dict[str, HealthCheckDetail]:
        names = list(check_factories)
        results = await asyncio.gather(*(check_factories[name]() for name in names))
        return dict(zip(names, results, strict=True))

    async def _instant_ok(self, message: str) -> HealthCheckDetail:
        return self._ok(message)

    def _list_qdrant_collections(self) -> None:
        from qdrant_client import QdrantClient  # noqa: PLC0415

        client = QdrantClient(
            url=self._app_settings.QDRANT_BASE_URL,
            api_key=self._app_settings.QDRANT_API_KEY,
            https=self._app_settings.QDRANT_HTTPS,
        )
        client.get_collections()

    def _ensure_qdrant_schema(self) -> None:
        from cadence_md.app.qdrant import get_qdrant_manager_from_settings  # noqa: PLC0415

        embedder = SimpleNamespace(model=self._app_settings.rag_config.embedding.model_name)
        qdrant_manager = get_qdrant_manager_from_settings(
            embedder=embedder,
            app_settings=self._app_settings,
        )
        qdrant_manager.ensure_collection_exists_and_schema_matches()

    def _models_url(self) -> str:
        base_url = str(self._app_settings.MODEL_INFERENCE_BASE_URL).rstrip("/")
        return f"{base_url}/models"

    def _inference_headers(self) -> dict[str, str]:
        api_key = self._app_settings.MODEL_INFERENCE_API_KEY
        if not api_key:
            return {}
        return {"Authorization": f"Bearer {api_key}"}

    def _expected_model_names(self) -> set[str]:
        cfg = self._app_settings.rag_config
        return {
            cfg.embedding.model_name,
            cfg.reranker.model_name,
            cfg.llm.model_name,
        }

    @staticmethod
    def _extract_model_ids(payload: Any) -> set[str]:
        if not isinstance(payload, Mapping):
            return set()

        raw_models = payload.get("data")
        if not isinstance(raw_models, Iterable) or isinstance(raw_models, (str, bytes)):
            return set()

        model_ids: set[str] = set()
        for item in raw_models:
            if isinstance(item, Mapping):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id:
                    model_ids.add(model_id)
        return model_ids

    @staticmethod
    def _response(details: dict[str, HealthCheckDetail]) -> HealthResponse:
        status = _aggregate_status(details.values())
        return HealthResponse(
            status=status,
            checks={name: detail.status for name, detail in details.items()},
            details=details,
        )

    @staticmethod
    def _ok(
        message: str,
        *,
        started: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> HealthCheckDetail:
        return HealthCheckDetail(
            status=HealthStatus.OK,
            message=message,
            latency_ms=_elapsed_ms(started),
            details=details or {},
        )

    @staticmethod
    def _degraded(
        message: str,
        *,
        started: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> HealthCheckDetail:
        return HealthCheckDetail(
            status=HealthStatus.DEGRADED,
            message=message,
            latency_ms=_elapsed_ms(started),
            details=details or {},
        )

    @staticmethod
    def _unavailable(
        message: str,
        *,
        started: float | None = None,
        error_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> HealthCheckDetail:
        response_details = dict(details or {})
        if error_type:
            response_details["error_type"] = error_type
        return HealthCheckDetail(
            status=HealthStatus.UNAVAILABLE,
            message=message,
            latency_ms=_elapsed_ms(started),
            details=response_details,
        )

    @staticmethod
    def _log_unavailable(check_name: str, exc: Exception) -> None:
        logger.warning(
            "Health check failed",
            extra={"check": check_name, "error_type": type(exc).__name__},
        )


def _aggregate_status(details: Iterable[HealthCheckDetail]) -> HealthStatus:
    statuses = {detail.status for detail in details}
    if HealthStatus.UNAVAILABLE in statuses:
        return HealthStatus.UNAVAILABLE
    if HealthStatus.DEGRADED in statuses:
        return HealthStatus.DEGRADED
    return HealthStatus.OK


def _elapsed_ms(started: float | None) -> float | None:
    if started is None:
        return None
    return round((time.perf_counter() - started) * 1000, 3)
