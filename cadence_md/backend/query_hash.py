"""Query hashing for RAG request logs (aligned with ``cadence_md.app.rag._query_hash``)."""

from __future__ import annotations

import hashlib


def compute_query_hash(query: str) -> str:
    """First 16 hex chars of SHA-256 for log correlation (not a secrecy mechanism)."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
