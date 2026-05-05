"""OpenAPI contract tests for the backend skeleton."""

from __future__ import annotations

from typing import Any

from cadence_md.backend.main import create_app
from cadence_md.backend.schemas.limits import MAX_QUERY_LENGTH


def _openapi_schema() -> dict[str, Any]:
    return create_app().openapi()


def test_openapi_contains_versioned_contract_paths() -> None:
    schema = _openapi_schema()

    expected_paths = {
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/users/me",
        "/api/v1/chat/messages",
        "/api/v1/chat/messages/{request_id}",
        "/api/v1/chat/messages/{request_id}/cancel",
        "/api/v1/chat/messages/{request_id}/retry",
        "/api/v1/health/live",
        "/api/v1/health/ready",
        "/api/v1/health/rag",
    }

    assert expected_paths <= set(schema["paths"])


def test_openapi_contains_auth_chat_health_and_error_schemas() -> None:
    schema = _openapi_schema()
    components = schema["components"]["schemas"]

    expected_components = {
        "RegisterRequest",
        "LoginRequest",
        "TokenResponse",
        "UserProfileResponse",
        "CreateRAGRequest",
        "RAGRequestStatus",
        "RAGRequestStatusResponse",
        "RAGAnswerResponse",
        "RAGSourceResponse",
        "ErrorResponse",
        "HealthResponse",
    }

    assert expected_components <= set(components)


def test_rag_request_status_enum_is_explicit() -> None:
    schema = _openapi_schema()

    assert schema["components"]["schemas"]["RAGRequestStatus"]["enum"] == [
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    ]


def test_create_rag_request_exposes_query_length_limit() -> None:
    schema = _openapi_schema()
    create_request = schema["components"]["schemas"]["CreateRAGRequest"]

    assert create_request["properties"]["query"]["maxLength"] == MAX_QUERY_LENGTH
    assert create_request["properties"]["query"]["minLength"] == 1


def test_rate_limit_error_responses_are_documented() -> None:
    schema = _openapi_schema()
    paths = schema["paths"]

    assert "429" in paths["/api/v1/auth/register"]["post"]["responses"]
    assert "429" in paths["/api/v1/auth/login"]["post"]["responses"]
    assert "429" in paths["/api/v1/chat/messages"]["post"]["responses"]
    assert "503" in paths["/api/v1/chat/messages"]["post"]["responses"]
