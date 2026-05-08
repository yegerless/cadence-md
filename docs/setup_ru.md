# Настройка dev-окружения CADENCE-MD

Эта инструкция описывает локальный dev stack CADENCE-MD: Python backend, RAG
worker, React frontend, Qdrant, Postgres, Redis, загрузку DVC-корпуса,
observability, тесты и внешний inference server.

Опции CLI и примеры команд вынесены в [`commands_ru.md`](commands_ru.md).

## Требования

- Python `>=3.13,<3.15`
- Poetry `>=2.0`
- Docker с Docker Compose
- Node.js и npm для локальной frontend-разработки
- DVC credentials для S3-compatible Yandex Cloud remote
- Внешний OpenAI-compatible inference server для embeddings, reranking и LLM
  generation

Dev compose file не запускает inference server. Запустите его отдельно и укажите
адрес в `MODEL_INFERENCE_BASE_URL`.

## 1. Клонирование репозитория

```bash
git clone git@github.com:yegerless/cadence-md.git
cd cadence-md
```

## 2. Установка Python-зависимостей

Установите зависимости приложения и dev-зависимости:

```bash
poetry install --with dev
```

Установите pre-commit hooks:

```bash
poetry run pre-commit install
```

Перед коммитом запускайте все настроенные hooks:

```bash
poetry run pre-commit run -av
```

Hooks включают Poetry validation, Ruff check/format, стандартные file checks,
Prettier для поддерживаемых текстовых форматов и codespell.

## 3. Переменные окружения

Создайте локальный env-файл из шаблона:

```bash
cp .env.example .env.dev
```

Замените placeholder-значения в `.env.dev`. Не коммитьте `.env.dev` и реальные
секреты.

Основные группы переменных в `.env.example`:

| Группа        | Переменные                                                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Backend/API   | `BACKEND_HOST`, `BACKEND_PORT`, `JWT_*`, `AUTH_*`, `CHAT_RATE_LIMIT_USER`, `GLOBAL_RAG_QUEUE_MAX`, `HEALTH_CHECK_TIMEOUT_SECONDS`         |
| Postgres      | `POSTGRES_*`, `DATABASE_URL`                                                                                                              |
| Redis/Celery  | `REDIS_*`, `CELERY_*`                                                                                                                     |
| Qdrant        | `QDRANT_BASE_URL`, `QDRANT__SERVICE__API_KEY`, `QDRANT_HTTPS`                                                                             |
| Inference     | `MODEL_INFERENCE_BASE_URL`, `MODEL_INFERENCE_API_KEY`                                                                                     |
| Observability | `LOG_*`, `PROMETHEUS_*`, `WORKER_METRICS_PORT`, `GRAFANA_*`                                                                               |
| Langfuse      | `LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `LANGFUSE_TRACE_QUERY_MODE`, `LANGFUSE_PROMPT_VERSION` |
| Corpus/DVC    | `S3_KEY_ID`, `S3_KEY`, `RAG_CORPUS_DIR`, `RAG_CONFIG__QDRANT_CONFIG__DATA_DIR`                                                            |
| QA generation | `GIGACHAT_API_KEY`, `GIGACHAT_MIN_INTERVAL_SEC`                                                                                           |
| Frontend      | `FRONTEND_PORT`, `VITE_API_BASE_URL`, `VITE_API_PROXY_TARGET`                                                                             |

Langfuse выключен по умолчанию. Если включаете его, храните реальные ключи
только в `.env.dev`. Медицинский текст запроса редактируется, пока явно не задан
`LANGFUSE_TRACE_QUERY_MODE=full` для разрешённой отладки.

В `.env.example` одновременно есть `GRAFANA_PORT=3000` и
`LANGFUSE_HOST=http://localhost:3000`. Если локальный Langfuse на хосте и
Grafana из compose работают одновременно, используйте разные порты или обновите
`LANGFUSE_HOST`.

## 4. Настройка и загрузка DVC-корпуса

Корпус клинических рекомендаций отслеживается через DVC. В `.dvc/config`
настроен Yandex Cloud S3-compatible remote:

- remote name: `yandex`
- bucket URL: `s3://clinical-recs`
- endpoint: `https://storage.yandexcloud.net/`

Для локального DVC на хосте настройте credentials:

```bash
poetry run dvc remote modify --local yandex access_key_id '<your_access_key_id>'
poetry run dvc remote modify --local yandex secret_access_key '<your_secret_access_key>'
poetry run dvc pull
```

Для Docker Compose положите те же credentials в `.env.dev` как `S3_KEY_ID` и
`S3_KEY`. Сервис `dvc-pull` скачивает `data.dvc` в общий volume `rag-corpus`.
`backend` и `rag-worker` монтируют этот volume read-only в `/app/data`.

## 5. Запуск dev stack

Проверьте compose-конфигурацию:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml config
```

Запустите полный dev stack:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up --build
```

В stack входят:

- `postgres`: база приложения
- `redis`: Celery broker/result backend
- `qdrant`: vector store
- `dvc-pull`: one-shot загрузка корпуса в `rag-corpus`
- `migrations`: `alembic upgrade head`
- `backend`: `uvicorn cadence_md.backend.main:app`
- `rag-worker`: Celery worker для RAG queue
- `frontend`: Vite dev server
- `prometheus` и `grafana`: локальная observability

Основной dev flow RAG асинхронный: backend принимает chat-запросы, а
`rag-worker` их обрабатывает. Если графу нужно уточнение врача, запрос переходит
в `awaiting_clarification`; frontend показывает форму уточнения и продолжает тот
же запрос через `POST /api/v1/chat/messages/{request_id}/clarification`.

Интерактивного пользовательского RAG CLI нет.

## 6. Частичные Compose-команды

Запустить только инфраструктурные зависимости:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up postgres redis qdrant
```

Запустить миграции вручную:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml run --rm migrations
```

Запустить только backend. Compose дождётся migrations, Redis, Qdrant и загрузки
DVC-корпуса согласно зависимостям сервисов:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up backend
```

Запустить RAG worker:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up rag-worker
```

Запустить frontend:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up frontend
```

Запустить observability services:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up prometheus grafana
```

Для синхронизации кода с контейнерами используйте Compose watch:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml watch backend rag-worker frontend
```

Правила `develop.watch`:

- `backend` и `rag-worker`: `sync+restart` для `./cadence_md`, rebuild для
  `pyproject.toml` и `poetry.lock`.
- `frontend`: sync `./frontend` в `/app`, ignore для `node_modules`, `dist` и
  `.vite`, rebuild при изменении `frontend/package*.json`.

## 7. Health checks и observability

Health endpoints:

```bash
curl -fsS http://127.0.0.1:8000/api/v1/health/live
curl -fsS http://127.0.0.1:8000/api/v1/health/ready
curl -fsS http://127.0.0.1:8000/api/v1/health/rag
docker compose --env-file .env.dev -f docker-compose-dev.yml exec rag-worker \
  python -m cadence_md.workers.rag_health --timeout 5
```

Metrics endpoints:

```bash
curl -fsS http://127.0.0.1:8000/metrics
curl -fsS http://127.0.0.1:9100/metrics
```

Локальные observability URL:

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`
- Метрики backend: `http://127.0.0.1:8000/metrics`
- Метрики worker: `http://127.0.0.1:9100/metrics`

RAG graph экспортирует latency узлов `query_rewrite`, `qdrant`, `rerank`,
`context`, `context_relevance`, `llm`, `answer_format`. Fallback/truncation
counters включают `query_rewrite_fallback`, `context_relevance_fallback`,
`answer_format_fallback`, `clarification_required`, retrieval fallback и
generation fallback.

## 8. Frontend-разработка

Установите зависимости:

```bash
cd frontend
npm install
```

Запуск с backend на хосте:

```bash
VITE_API_BASE_URL= VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
```

Build и тесты:

```bash
npm test
npm run lint
npm run build
```

По умолчанию `VITE_API_BASE_URL` пустой, а Vite dev server проксирует
`/api/v1/...` в `VITE_API_PROXY_TARGET`. В Compose target по умолчанию равен
`http://backend:8000`.

Frontend auth хранит только короткоживущий access token в `sessionStorage`.
Logout и любой API `401` очищают client auth state и переводят пользователя на
`/login`.

Краткая справка по frontend находится в
[`../frontend/README_RU.md`](../frontend/README_RU.md).

## 9. Backend tests и quality checks

Быстрые unit-тесты:

```bash
poetry run pytest -m "not integration"
```

Ruff checks:

```bash
poetry run ruff check cadence_md tests metrics commands.py
poetry run ruff format --check cadence_md tests metrics commands.py
```

Pre-commit:

```bash
poetry run pre-commit run -av
```

## 10. Integration test stack

Поднимите отдельные зависимости для integration-тестов. Тесты сами сервисы не
стартуют:

```bash
docker compose -f docker-compose-test.yml up -d postgres-test redis-test
```

В `docker-compose-test.yml` также есть optional `qdrant-test` за profile
`with-qdrant`:

```bash
docker compose -f docker-compose-test.yml --profile with-qdrant up -d qdrant-test
```

Проверьте compose files:

```bash
docker compose -f docker-compose-test.yml config
docker compose --env-file .env.example -f docker-compose-dev.yml config
```

Запустите integration tests:

```bash
INTEGRATION_DATABASE_URL=postgresql+asyncpg://cadence_md:cadence_md_dev@127.0.0.1:55432/cadence_md_test \
INTEGRATION_REDIS_URL=redis://127.0.0.1:56379/0 \
poetry run pytest -m integration
```

## 11. CLI-команды

Запускайте проектный CLI через Poetry:

```bash
poetry run python commands.py --help
poetry run python commands.py parse-pdf --help
poetry run python commands.py generate-qa --help
poetry run python commands.py metrics-eval-full --help
poetry run python commands.py metrics-eval-retriever --help
```

Полный CLI-справочник находится в [`commands_ru.md`](commands_ru.md).

## 12. Внешний inference server

CADENCE-MD сейчас использует один OpenAI-compatible base URL
(`MODEL_INFERENCE_BASE_URL`) для всех model calls:

- embedding model
- reranker model
- LLM для генерации ответа

На macOS рекомендуемый локальный вариант для GGUF-моделей — `llama.cpp`. На
Linux для GPU-serving обычно удобнее vLLM.

Установите `llama.cpp` на macOS:

```bash
brew install llama.cpp
```

Скачайте GGUF-файлы через `llama-cli`:

```bash
llama-cli --hf-repo repo-owner/hf-repo --hf-file filename.gguf
```

Пример запуска `llama-server`:

```bash
llama-server \
  --models-dir /path/to/models \
  --models-preset /path/to/models.ini \
  --models-max 3 \
  --metrics \
  --perf \
  --log-timestamps \
  --log-prefix
```

Имена моделей задаются в `llama-configs/*.ini`. Синхронизируйте эти имена с RAG
settings приложения. После запуска сервера задайте:

```bash
MODEL_INFERENCE_BASE_URL=http://host.docker.internal:8080/v1
MODEL_INFERENCE_API_KEY=change-me-local-only
```

Для host-only backend запуска используйте `http://127.0.0.1:8080/v1` вместо
`host.docker.internal`.
