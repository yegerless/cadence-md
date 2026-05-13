"""Celery application configuration for CADENCE-MD workers."""

from __future__ import annotations

from celery import Celery
from kombu import Queue

from cadence_md.backend.settings import backend_settings


def create_celery_app() -> Celery:
    """Create a Celery app using backend-owned broker/result settings."""
    app = Celery(
        "cadence_md",
        broker=backend_settings.CELERY_BROKER_URL,
        backend=backend_settings.CELERY_RESULT_BACKEND,
        include=("cadence_md.workers.rag_tasks",),
    )
    app.conf.update(
        accept_content=("json",),
        result_serializer="json",
        task_serializer="json",
        task_default_queue=backend_settings.CELERY_RAG_QUEUE_NAME,
        task_queues=(Queue(backend_settings.CELERY_RAG_QUEUE_NAME),),
        task_routes={
            backend_settings.CELERY_RAG_TASK_NAME: {
                "queue": backend_settings.CELERY_RAG_QUEUE_NAME,
            }
        },
        task_soft_time_limit=backend_settings.CELERY_TASK_SOFT_TIME_LIMIT_SECONDS,
        task_time_limit=backend_settings.CELERY_TASK_TIME_LIMIT_SECONDS,
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = create_celery_app()
