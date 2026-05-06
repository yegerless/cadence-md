"""Command-line health probe for the Celery RAG worker."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from typing import Any

from celery import Celery

from cadence_md.workers.celery_app import celery_app


def default_worker_destination() -> str:
    """Return the Celery node name for the current container hostname."""
    return os.getenv("CELERY_WORKER_HEALTH_DESTINATION") or f"celery@{socket.gethostname()}"


def worker_ping_ok(
    app: Celery = celery_app,
    *,
    destination: str | None = None,
    timeout_seconds: float = 5.0,
) -> bool:
    """Return True when the target Celery worker responds to inspect ping."""
    destinations = [destination] if destination else None
    replies = app.control.ping(destination=destinations, timeout=timeout_seconds)
    return _has_pong(replies, destination=destination)


def _has_pong(replies: Any, *, destination: str | None) -> bool:
    if not isinstance(replies, list) or not replies:
        return False

    for reply in replies:
        if not isinstance(reply, dict):
            continue
        for worker_name, payload in reply.items():
            if destination is not None and worker_name != destination:
                continue
            if isinstance(payload, dict) and payload.get("ok") == "pong":
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Run the worker health probe and return a process exit code."""
    parser = argparse.ArgumentParser(description="Check that the RAG Celery worker responds.")
    parser.add_argument(
        "--destination",
        default=default_worker_destination(),
        help="Celery worker node name to ping. Defaults to celery@$HOSTNAME.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Celery inspect ping timeout in seconds.",
    )
    args = parser.parse_args(argv)

    try:
        ok = worker_ping_ok(destination=args.destination, timeout_seconds=args.timeout)
    except Exception as exc:  # pragma: no cover - exercised through non-zero CLI behavior.
        sys.stderr.write(f"RAG worker healthcheck failed: {type(exc).__name__}\n")
        return 1

    if ok:
        sys.stdout.write("RAG worker is healthy\n")
        return 0

    sys.stderr.write("RAG worker did not respond to health ping\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
