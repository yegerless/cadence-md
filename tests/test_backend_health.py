"""Tests for backend health and readiness endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from cadence_md.backend.api.deps import get_health_service
from cadence_md.backend.main import create_app
from cadence_md.backend.schemas.health import HealthCheckDetail, HealthResponse, HealthStatus
from cadence_md.backend.services.health import HealthService
from cadence_md.backend.settings import BackendSettings
from cadence_md.workers.rag_health import worker_ping_ok


def _health_response(status: HealthStatus, checks: dict[str, HealthStatus]) -> HealthResponse:
    return HealthResponse(
        status=status,
        checks=checks,
        details={
            name: HealthCheckDetail(status=check_status, message=f"{name} status")
            for name, check_status in checks.items()
        },
    )


class FakeHealthService:
    """Simple health service test double for route-level tests."""

    def __init__(self, *, ready: HealthResponse, rag: HealthResponse) -> None:
        self.ready_response = ready
        self.rag_response = rag
        self.ready_calls = 0
        self.rag_calls = 0

    async def ready(self) -> HealthResponse:
        self.ready_calls += 1
        return self.ready_response

    async def rag(self) -> HealthResponse:
        self.rag_calls += 1
        return self.rag_response


@pytest.mark.asyncio
async def test_live_returns_ok_without_health_service() -> None:
    app = create_app()

    def fail_if_called() -> FakeHealthService:
        raise AssertionError("health service must not be constructed for /live")

    app.dependency_overrides[get_health_service] = fail_if_called
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["checks"] == {"backend": "ok"}


@pytest.mark.asyncio
async def test_ready_returns_ok_when_dependencies_are_ready() -> None:
    service = FakeHealthService(
        ready=_health_response(
            HealthStatus.OK,
            {
                "backend": HealthStatus.OK,
                "postgres": HealthStatus.OK,
                "redis": HealthStatus.OK,
                "qdrant": HealthStatus.OK,
                "qdrant_schema": HealthStatus.OK,
                "inference": HealthStatus.OK,
            },
        ),
        rag=_health_response(HealthStatus.OK, {"qdrant": HealthStatus.OK}),
    )
    app = create_app()
    app.dependency_overrides[get_health_service] = lambda: service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"]["postgres"] == "ok"
    assert service.ready_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_check",
    ["postgres", "redis", "qdrant_schema", "inference"],
)
async def test_ready_returns_503_when_dependency_is_unavailable(failed_check: str) -> None:
    checks = {
        "backend": HealthStatus.OK,
        "postgres": HealthStatus.OK,
        "redis": HealthStatus.OK,
        "qdrant": HealthStatus.OK,
        "qdrant_schema": HealthStatus.OK,
        "inference": HealthStatus.OK,
    }
    checks[failed_check] = HealthStatus.UNAVAILABLE
    app = create_app()
    app.dependency_overrides[get_health_service] = lambda: FakeHealthService(
        ready=_health_response(HealthStatus.UNAVAILABLE, checks),
        rag=_health_response(HealthStatus.OK, {"qdrant": HealthStatus.OK}),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"][failed_check] == "unavailable"


@pytest.mark.asyncio
async def test_rag_returns_details_without_secrets() -> None:
    app = create_app()
    app.dependency_overrides[get_health_service] = lambda: FakeHealthService(
        ready=_health_response(HealthStatus.OK, {"backend": HealthStatus.OK}),
        rag=HealthResponse(
            status=HealthStatus.UNAVAILABLE,
            checks={
                "qdrant": HealthStatus.OK,
                "inference": HealthStatus.UNAVAILABLE,
                "queue": HealthStatus.OK,
            },
            details={
                "qdrant": HealthCheckDetail(status=HealthStatus.OK, message="Qdrant is reachable."),
                "inference": HealthCheckDetail(
                    status=HealthStatus.UNAVAILABLE,
                    message="Inference API is unreachable.",
                    details={"error_type": "ConnectError"},
                ),
                "queue": HealthCheckDetail(status=HealthStatus.OK, message="Queue is reachable."),
            },
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health/rag")

    assert response.status_code == 503
    body_text = response.text
    assert "lm-studio" not in body_text
    assert "api_key" not in body_text.lower()
    assert response.json()["details"]["inference"]["details"] == {"error_type": "ConnectError"}


@pytest.mark.asyncio
async def test_inference_models_check_accepts_all_required_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAsyncClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            pass

        async def get(self, _: str, *, headers: dict[str, str]) -> httpx.Response:
            assert headers == {"Authorization": "Bearer test-key"}
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "qwen3-embedding-4b"},
                        {"id": "qwen3-reranker-4b"},
                        {"id": "qwen3.5-9b"},
                    ]
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    service = HealthService(
        backend_settings=BackendSettings(HEALTH_CHECK_TIMEOUT_SECONDS=0.1),
        app_settings=_fake_app_settings(api_key="test-key"),
    )

    result = await service.check_inference_models()

    assert result.status == HealthStatus.OK
    assert result.details["models_checked"] == [
        "qwen3-embedding-4b",
        "qwen3-reranker-4b",
        "qwen3.5-9b",
    ]


@pytest.mark.asyncio
async def test_inference_models_check_marks_missing_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAsyncClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            pass

        async def get(self, _: str, *, headers: dict[str, str]) -> httpx.Response:
            assert headers == {}
            return httpx.Response(200, json={"data": [{"id": "qwen3-embedding-4b"}]})

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    service = HealthService(
        backend_settings=BackendSettings(HEALTH_CHECK_TIMEOUT_SECONDS=0.1),
        app_settings=_fake_app_settings(api_key=""),
    )

    result = await service.check_inference_models()

    assert result.status == HealthStatus.UNAVAILABLE
    assert result.details["missing_models"] == ["qwen3-reranker-4b", "qwen3.5-9b"]


def test_worker_ping_ok_accepts_target_pong() -> None:
    app = SimpleNamespace(
        control=SimpleNamespace(
            ping=lambda destination, timeout: [{"celery@test-host": {"ok": "pong"}}]
        )
    )

    assert worker_ping_ok(app, destination="celery@test-host", timeout_seconds=1.0)


def test_worker_ping_ok_rejects_empty_response() -> None:
    app = SimpleNamespace(control=SimpleNamespace(ping=lambda destination, timeout: []))

    assert not worker_ping_ok(app, destination="celery@test-host", timeout_seconds=1.0)


def _fake_app_settings(*, api_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        MODEL_INFERENCE_BASE_URL="http://inference.test/v1",
        MODEL_INFERENCE_API_KEY=api_key,
        rag_config=SimpleNamespace(
            embedding=SimpleNamespace(model_name="qwen3-embedding-4b"),
            reranker=SimpleNamespace(model_name="qwen3-reranker-4b"),
            llm=SimpleNamespace(model_name="qwen3.5-9b"),
        ),
    )
