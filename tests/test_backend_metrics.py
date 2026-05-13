"""Smoke tests for backend Prometheus metrics."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from cadence_md.backend.main import create_app
from cadence_md.observability.settings import observability_settings


@pytest.fixture(autouse=True)
def _enable_prometheus_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep metrics smoke tests independent from local .env.dev overrides."""
    monkeypatch.setattr(observability_settings, "PROMETHEUS_ENABLED", True)


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_text() -> None:
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/api/v1/health/live")
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "cadence_backend_requests_total" in response.text
    assert "/api/v1/health/live" in response.text


@pytest.mark.asyncio
async def test_metrics_endpoint_is_not_versioned_or_authenticated() -> None:
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert "Authentication is required" not in response.text
