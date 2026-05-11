# CADENCE-MD Development Setup

This guide describes the local development stack for CADENCE-MD: Python backend,
RAG worker, React frontend, Qdrant, Postgres, Redis, DVC corpus loading,
observability, tests, and the external inference server.

For CLI options and examples, see [`commands.md`](commands.md).

## Prerequisites

- Python `>=3.13,<3.15`
- Poetry `>=2.0`
- Docker with Docker Compose
- Node.js and npm for local frontend development
- DVC credentials for the S3-compatible Yandex Cloud remote
- An external OpenAI-compatible inference server for embeddings, reranking, and
  LLM generation

The dev compose file does not start an inference server. Start it separately and
point `MODEL_INFERENCE_BASE_URL` at it.

## 1. Clone The Repository

```bash
git clone git@github.com:yegerless/cadence-md.git
cd cadence-md
```

## 2. Install Python Dependencies

Install application and development dependencies:

```bash
poetry install --with dev
```

Install pre-commit hooks:

```bash
poetry run pre-commit install
```

Run all configured hooks before committing:

```bash
poetry run pre-commit run -av
```

The hooks include Poetry validation, Ruff check/format, standard file checks,
Prettier for supported text formats, and codespell.

## 3. Configure Environment Variables

Create a local environment file from the committed template:

```bash
cp .env.example .env.dev
```

Replace placeholder values in `.env.dev`. Never commit `.env.dev` or real
secrets.

Important groups in `.env.example`:

| Group         | Variables                                                                                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Backend/API   | `BACKEND_HOST`, `BACKEND_PORT`, `JWT_*`, `AUTH_*`, `CHAT_RATE_LIMIT_USER`, `GLOBAL_RAG_QUEUE_MAX`, `HEALTH_CHECK_TIMEOUT_SECONDS`                              |
| Postgres      | `POSTGRES_*`, `DATABASE_URL`                                                                                                                                   |
| Redis/Celery  | `REDIS_*`, `CELERY_*`                                                                                                                                          |
| Qdrant        | `QDRANT_BASE_URL`, `QDRANT__SERVICE__API_KEY`, `QDRANT_HTTPS`                                                                                                  |
| Inference     | `MODEL_INFERENCE_BASE_URL`, `MODEL_INFERENCE_API_KEY`                                                                                                          |
| Observability | `LOG_*`, `PROMETHEUS_*`, `WORKER_METRICS_PORT`, `GRAFANA_*`                                                                                                    |
| Langfuse      | `LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `LANGFUSE_HOST`, `LANGFUSE_TRACE_QUERY_MODE`, `LANGFUSE_PROMPT_VERSION` |
| Corpus/DVC    | `S3_KEY_ID`, `S3_KEY`, `RAG_CORPUS_DIR`, `RAG_CONFIG__QDRANT_CONFIG__DATA_DIR`                                                                                 |
| QA generation | `GIGACHAT_API_KEY`, `GIGACHAT_MIN_INTERVAL_SEC`, `GIGACHAT_EMBEDDINGS_MAX_*`                                                                                   |
| Frontend      | `FRONTEND_PORT`, `VITE_API_BASE_URL`, `VITE_API_PROXY_TARGET`                                                                                                  |

Langfuse is disabled by default. If you enable it, keep real keys only in
`.env.dev`. Medical query text is redacted unless
`LANGFUSE_TRACE_QUERY_MODE=full` is explicitly configured for approved
debugging.

Note that `.env.example` uses `GRAFANA_PORT=3000` and
`LANGFUSE_BASE_URL=http://localhost:3000`. If you run Langfuse locally on the
host and Grafana through compose at the same time, use different ports or update
`LANGFUSE_BASE_URL`. For Langfuse Cloud, use a full URL with scheme, for example
`https://cloud.langfuse.com`.

## 4. Configure And Download The DVC Corpus

The clinical guideline corpus is tracked with DVC. The committed `.dvc/config`
uses a Yandex Cloud S3-compatible remote:

- remote name: `yandex`
- bucket URL: `s3://clinical-recs`
- endpoint: `https://storage.yandexcloud.net/`

For host-local DVC usage, configure credentials locally:

```bash
poetry run dvc remote modify --local yandex access_key_id '<your_access_key_id>'
poetry run dvc remote modify --local yandex secret_access_key '<your_secret_access_key>'
poetry run dvc pull
```

For Docker Compose usage, put the same credentials in `.env.dev` as `S3_KEY_ID`
and `S3_KEY`. The `dvc-pull` service downloads `data.dvc` into the shared
`rag-corpus` volume. `backend` and `rag-worker` mount that volume read-only at
`/app/data`.

## 5. Run The Dev Stack

Validate the compose configuration:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml config
```

Start the full development stack:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up --build
```

The stack includes:

- `postgres`: application database
- `redis`: Celery broker/result backend
- `qdrant`: vector store
- `dvc-pull`: one-shot corpus download into `rag-corpus`
- `migrations`: `alembic upgrade head`
- `backend`: `uvicorn cadence_md.backend.main:app`
- `rag-worker`: Celery worker for the RAG queue
- `frontend`: Vite dev server
- `prometheus` and `grafana`: local observability

The primary dev RAG flow is asynchronous: the backend accepts chat requests and
`rag-worker` processes them. If the graph needs a human clarification, the
request moves to `awaiting_clarification`; the frontend shows the clarification
form and resumes the same request through
`POST /api/v1/chat/messages/{request_id}/clarification`.

Chat conversations are persisted per user. Clients can use
`POST /api/v1/chat/conversations` to create a chat,
`GET /api/v1/chat/conversations` to list active chats,
`GET /api/v1/chat/conversations/{chat_id}/messages` to restore message history,
and `DELETE /api/v1/chat/conversations/{chat_id}` to soft-delete a chat. Sending
`chat_id` with `POST /api/v1/chat/messages` binds a RAG request to that chat; if
it is omitted, the backend creates a new chat. Soft delete hides the chat from
the UI and active chat APIs without physically deleting `rag_requests` or
`rag_responses`.

RAG validation stays separate from user chat history: the metrics pipeline runs
QA cases through `RAGService.run` / `RAGService.retrieve` with clarification
disabled for batch evaluation.

There is no interactive user-facing RAG CLI.

## 6. Partial Compose Commands

Start only infrastructure dependencies:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up postgres redis qdrant
```

Run migrations manually:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml run --rm migrations
```

Start backend only. Compose waits for migrations, Redis, Qdrant, and DVC corpus
loading according to service dependencies:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up backend
```

Start the RAG worker:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up rag-worker
```

Start frontend:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up frontend
```

Start observability services:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up prometheus grafana
```

Use Compose watch for container code sync:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml watch backend rag-worker frontend
```

`develop.watch` rules:

- `backend` and `rag-worker`: `sync+restart` for `./cadence_md`, rebuild for
  `pyproject.toml` and `poetry.lock`.
- `frontend`: sync `./frontend` to `/app`, ignore `node_modules`, `dist`, and
  `.vite`, rebuild on `frontend/package*.json`.

## 7. Health Checks And Observability

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

Local observability URLs:

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`
- Backend metrics: `http://127.0.0.1:8000/metrics`
- Worker metrics: `http://127.0.0.1:9100/metrics`

The RAG graph exports node latency for `query_rewrite`, `qdrant`, `rerank`,
`context`, `context_relevance`, `llm`, and `answer_format`. Fallback/truncation
counters include `query_rewrite_fallback`, `context_relevance_fallback`,
`answer_format_fallback`, `clarification_required`, retrieval fallback, and
generation fallback.

## 8. Frontend Development

Install dependencies:

```bash
cd frontend
npm install
```

Run with a backend on the host:

```bash
VITE_API_BASE_URL= VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
```

Build and test:

```bash
npm test
npm run lint
npm run build
```

By default, `VITE_API_BASE_URL` is empty and the Vite dev server proxies
`/api/v1/...` to `VITE_API_PROXY_TARGET`. In Compose, the target defaults to
`http://backend:8000`.

Frontend auth stores only the short-lived access token in `sessionStorage`.
Logout and any API `401` clear client auth state and redirect to `/login`.

See [`../frontend/README.md`](../frontend/README.md) for the short frontend
reference.

## 9. Backend Tests And Quality Checks

Fast unit tests:

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

## 10. Integration Test Stack

Bring up dedicated integration dependencies. Tests do not auto-start services:

```bash
docker compose -f docker-compose-test.yml up -d postgres-test redis-test
```

`docker-compose-test.yml` also has optional `qdrant-test` behind the
`with-qdrant` profile:

```bash
docker compose -f docker-compose-test.yml --profile with-qdrant up -d qdrant-test
```

Validate compose files:

```bash
docker compose -f docker-compose-test.yml config
docker compose --env-file .env.example -f docker-compose-dev.yml config
```

Run integration tests:

```bash
INTEGRATION_DATABASE_URL=postgresql+asyncpg://cadence_md:cadence_md_dev@127.0.0.1:55432/cadence_md_test \
INTEGRATION_REDIS_URL=redis://127.0.0.1:56379/0 \
poetry run pytest -m integration
```

## 11. CLI Commands

Use the project CLI through Poetry:

```bash
poetry run python commands.py --help
poetry run python commands.py parse-pdf --help
poetry run python commands.py generate-qa --help
poetry run python commands.py metrics-eval-full --help
poetry run python commands.py metrics-eval-retriever --help
```

The full command reference lives in [`commands.md`](commands.md).

## 12. External Inference Server

CADENCE-MD currently uses one OpenAI-compatible base URL
(`MODEL_INFERENCE_BASE_URL`) for all model calls:

- embedding model
- reranker model
- LLM for answer generation

On macOS, `llama.cpp` is the recommended local option for GGUF models. On Linux,
vLLM is usually a better fit for GPU-backed serving.

Install `llama.cpp` on macOS:

```bash
brew install llama.cpp
```

Download GGUF files with `llama-cli`:

```bash
llama-cli --hf-repo repo-owner/hf-repo --hf-file filename.gguf
```

Example `llama-server` startup:

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

Model names are configured in `llama-configs/*.ini`. Match those names with the
RAG settings used by the app. Once the server is running, set:

```bash
MODEL_INFERENCE_BASE_URL=http://host.docker.internal:8080/v1
MODEL_INFERENCE_API_KEY=change-me-local-only
```

For a host-only backend run, use `http://127.0.0.1:8080/v1` instead of
`host.docker.internal`.
