"""Privacy helpers for logs and third-party observability payloads."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, Literal

SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "token",
    "jwt",
    "authorization",
    "api_key",
    "apikey",
    "secret",
)
EMAIL_RE = re.compile(r"(?P<first>[^@\s]{1})[^@\s]*@(?P<domain>[^@\s]+)")


def query_hash_only(query: str) -> str:
    """Return the stable short hash used to correlate medical queries without logging text."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def mask_email(value: str) -> str:
    """Mask an email address while preserving a small amount of debugging context."""
    return EMAIL_RE.sub(lambda match: f"{match.group('first')}***@{match.group('domain')}", value)


def redact_medical_query(query: str, mode: Literal["redacted", "hash", "full"] = "redacted") -> str:
    """Apply the configured medical-query redaction policy."""
    if mode == "full":
        return query
    qh = query_hash_only(query)
    if mode == "hash":
        return qh
    return f"[redacted medical query: {qh}]"


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def redact_value(key: str, value: Any) -> Any:
    """Redact a single value based on key semantics and lightweight value patterns."""
    if _is_sensitive_key(key):
        redacted: Any = "[redacted]"
    elif value is None:
        redacted = None
    elif isinstance(value, str):
        redacted = mask_email(value) if "@" in value else value
    elif isinstance(value, Mapping):
        redacted = redact_mapping(value)
    elif isinstance(value, list):
        redacted = [redact_value(key, item) for item in value]
    elif isinstance(value, tuple):
        redacted = tuple(redact_value(key, item) for item in value)
    else:
        redacted = value
    return redacted


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly copy with secrets and obvious PII removed."""
    return {key: redact_value(key, value) for key, value in data.items()}
