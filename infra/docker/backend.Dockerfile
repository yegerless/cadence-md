FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    "alembic>=1.18.4,<2.0.0" \
    "asyncpg>=0.31.0,<0.32.0" \
    "fastapi>=0.136.1,<0.137.0" \
    "greenlet>=3.5.0,<4.0.0" \
    "pydantic-settings>=2.0.0,<3.0.0" \
    "sqlalchemy>=2.0.49,<3.0.0" \
    "uvicorn>=0.46.0,<0.47.0"

COPY alembic.ini ./alembic.ini
COPY cadence_md ./cadence_md

EXPOSE 8000
