FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi>=0.136.1,<0.137.0" \
    "uvicorn>=0.46.0,<0.47.0"

COPY cadence_md ./cadence_md

EXPOSE 8000
