"""Tests for observability privacy guards and structured logging."""

from __future__ import annotations

import json
import logging

from cadence_md.observability.logging import JsonFormatter, log_context
from cadence_md.observability.privacy import redact_mapping, redact_medical_query


def test_redact_mapping_removes_secrets_and_masks_email() -> None:
    redacted = redact_mapping(
        {
            "password": "plain",
            "api_key": "secret",
            "Authorization": "Bearer jwt",
            "email": "doctor@example.org",
            "user_id": "user-1",
        }
    )

    assert redacted["password"] == "[redacted]"
    assert redacted["api_key"] == "[redacted]"
    assert redacted["Authorization"] == "[redacted]"
    assert redacted["email"] == "d***@example.org"
    assert redacted["user_id"] == "user-1"


def test_redact_medical_query_defaults_to_hash_marker() -> None:
    redacted = redact_medical_query("Пациент с гипертензией", "redacted")

    assert redacted.startswith("[redacted medical query: ")
    assert "гипертензией" not in redacted


def test_json_formatter_includes_context_and_redacts_extra() -> None:
    formatter = JsonFormatter()
    logger = logging.getLogger("test-json-formatter")

    with log_context(request_id="req-1", rag_request_id="rag-1", task_id="task-1"):
        record = logger.makeRecord(
            logger.name,
            logging.INFO,
            __file__,
            10,
            "message",
            args=(),
            exc_info=None,
            extra={
                "request_id": "req-1",
                "rag_request_id": "rag-1",
                "task_id": "task-1",
                "jwt": "secret",
                "email": "doctor@example.org",
            },
        )

    payload = json.loads(formatter.format(record))

    assert payload["request_id"] == "req-1"
    assert payload["rag_request_id"] == "rag-1"
    assert payload["task_id"] == "task-1"
    assert payload["jwt"] == "[redacted]"
    assert payload["email"] == "d***@example.org"
