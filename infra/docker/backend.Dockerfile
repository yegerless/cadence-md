FROM python:3.13-slim

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y build-essential curl zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-ansi

COPY data.dvc ./data.dvc
COPY .dvc/config ./.dvc/config
COPY alembic.ini ./alembic.ini
COPY cadence_md ./cadence_md

EXPOSE 8000
