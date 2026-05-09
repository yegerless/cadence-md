"""Structured API exceptions mapped to ``ErrorResponse``."""

from __future__ import annotations


class ApiError(Exception):
    """Raised for domain errors that must surface as ``ErrorResponse`` JSON."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
