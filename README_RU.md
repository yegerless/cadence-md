# CADENCE-MD

**Clinical Assistant for Diagnosis, Evidence Navigation & Case Evaluation for
Medical Doctors**

CADENCE-MD — RAG-ассистент для российских врачей. Проект ищет доказательные
фрагменты в клинических рекомендациях Минздрава РФ, принимает асинхронные
chat-запросы через FastAPI backend и Celery worker, а для пользователя
предоставляет React SPA.

Это исследовательский проект. Он не предназначен для прямого клинического
использования без валидации.

## Что входит в проект

- Retrieval по клиническим рекомендациям из DVC-корпуса PDF, индексированного в
  Qdrant.
- LangGraph RAG pipeline с optional query rewriting, context relevance grading,
  query clarification, answer formatting и контролируемыми fallback-сценариями.
- FastAPI backend с JWT auth, per-user историей чатов, async chat lifecycle,
  health checks, Prometheus metrics и optional Langfuse tracing.
- Celery/Redis worker для асинхронной генерации RAG-ответов.
- React + Vite frontend для login, registration, истории чатов, profile,
  уточнений, отображения источников и авторизованного скачивания PDF.
- Metrics pipeline для full RAG evaluation и retriever-only evaluation; batch
  validation не использует пользовательскую историю чатов.

## Стек

- Python 3.13, Poetry
- FastAPI, SQLAlchemy/Alembic, Postgres, Redis, Celery
- LangGraph, LangChain, Qdrant
- DVC + S3-compatible storage для корпуса клинических рекомендаций
- React 19, Vite, TypeScript, React Router, TanStack Query, Vitest
- Docker Compose dev stack с backend, worker, frontend, Qdrant, Postgres, Redis,
  migrations, Prometheus и Grafana
- Внешний OpenAI-compatible inference server для embeddings, reranking и LLM
  generation

## Документация

- Подробная dev-настройка: [`docs/setup_ru.md`](docs/setup_ru.md)
- Полный CLI-справочник: [`docs/commands_ru.md`](docs/commands_ru.md)
- Frontend: [`frontend/README_RU.md`](frontend/README_RU.md)
- English README: [`README.md`](README.md)

## Быстрый старт

Установите Python-зависимости:

```bash
poetry install --with dev
```

Создайте локальный env-файл и замените placeholder-значения. Не коммитьте
`.env.dev`.

```bash
cp .env.example .env.dev
```

Запустите внешний OpenAI-compatible inference server, затем проверьте и
поднимите dev stack:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml config
docker compose --env-file .env.dev -f docker-compose-dev.yml up --build
```

Для синхронизации кода с контейнерами во время разработки:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml watch backend rag-worker frontend
```

Основные локальные URL:

- Frontend: `http://127.0.0.1:5173`
- Backend API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health/ready`
- Метрики backend: `http://127.0.0.1:8000/metrics`
- Метрики worker: `http://127.0.0.1:9100/metrics`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

## CLI

Проектные команды доступны через [`commands.py`](commands.py):

```bash
poetry run python commands.py --help
poetry run python commands.py <subcommand> --help
```

Доступные подкоманды:

- `parse-pdf`: парсинг PDF клинических рекомендаций в JSONL с секциями.
- `generate-qa`: генерация синтетического QA-датасета из распарсенных секций.
- `metrics-eval-full`: генерация ответов, RAGAS и retrieval metrics.
- `metrics-eval-retriever`: оценка retrieve + rerank без генерации ответа.

Опции, примеры и форматы артефактов описаны в
[`docs/commands_ru.md`](docs/commands_ru.md).

Интерактивного пользовательского RAG CLI нет. Пользовательские запросы идут
через FastAPI chat API и асинхронно обрабатываются `rag-worker`.

## Docker Compose

Dev compose запускается из корня репозитория:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up --build
```

В stack входят:

- `postgres`, `redis`, `qdrant`
- `dvc-pull`, который скачивает `data.dvc` в общий volume `rag-corpus`
- `migrations` для `alembic upgrade head`
- `backend` с `uvicorn cadence_md.backend.main:app`
- `rag-worker` с Celery RAG queue
- `frontend` с Vite dev server
- `prometheus`, `grafana`

Inference server намеренно не входит в `docker-compose-dev.yml`. Адрес и ключ
задаются через `MODEL_INFERENCE_BASE_URL` и `MODEL_INFERENCE_API_KEY`.

## Frontend

Локальный запуск frontend:

```bash
npm --prefix frontend install
VITE_API_BASE_URL= VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm --prefix frontend run dev
```

Приложение обращается к `/api/v1`. Пустой `VITE_API_BASE_URL` включает
same-origin proxy Vite и не требует CORS в локальной разработке. Auth хранит
только короткоживущий access token в `sessionStorage`; logout и API `401`
очищают client state и требуют повторного входа.

## Проверки качества

Backend unit-тесты:

```bash
poetry run pytest -m "not integration"
```

Backend lint и format checks:

```bash
poetry run ruff check cadence_md tests metrics commands.py
poetry run ruff format --check cadence_md tests metrics commands.py
```

Frontend checks:

```bash
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run build
```

Зависимости для integration-тестов описаны в `docker-compose-test.yml`:

```bash
docker compose -f docker-compose-test.yml up -d postgres-test redis-test
INTEGRATION_DATABASE_URL=postgresql+asyncpg://cadence_md:cadence_md_dev@127.0.0.1:55432/cadence_md_test \
INTEGRATION_REDIS_URL=redis://127.0.0.1:56379/0 \
poetry run pytest -m integration
```

## Лицензия

Проприетарная - см. [`LICENSE_RU.md`](LICENSE_RU.md).
