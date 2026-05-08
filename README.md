# CADENCE-MD

**Clinical Assistant for Diagnosis, Evidence Navigation & Case Evaluation for
Medical Doctors**

CADENCE-MD is a RAG-based medical assistant for Russian physicians. It retrieves
evidence from Russian Ministry of Health clinical guidelines, serves
asynchronous chat requests through a FastAPI backend and Celery worker, and
provides a React SPA for the clinical workflow.

This is a research project and is not intended for direct clinical use without
validation.

## What Is Included

- Clinical guideline retrieval over a DVC-tracked PDF corpus stored in Qdrant.
- LangGraph RAG pipeline with optional query rewriting, context relevance
  grading, query clarification, answer formatting, and structured fallbacks.
- FastAPI backend with JWT auth, async chat request lifecycle, health checks,
  Prometheus metrics, and optional Langfuse tracing.
- Celery/Redis RAG worker for asynchronous answer generation.
- React + Vite frontend for login, registration, chat, profile, clarification
  handling, source rendering, and authenticated PDF downloads.
- Metrics pipeline for full RAG evaluation and retriever-only evaluation.

## Stack

- Python 3.13, Poetry
- FastAPI, SQLAlchemy/Alembic, Postgres, Redis, Celery
- LangGraph, LangChain, Qdrant
- DVC + S3-compatible storage for the clinical guideline corpus
- React 19, Vite, TypeScript, React Router, TanStack Query, Vitest
- Docker Compose dev stack with backend, worker, frontend, Qdrant, Postgres,
  Redis, migrations, Prometheus, and Grafana
- External OpenAI-compatible inference server for embeddings, reranking, and LLM
  generation

## Documentation

- Detailed development setup: [`docs/setup.md`](docs/setup.md)
- Full CLI reference: [`docs/commands.md`](docs/commands.md)
- Frontend notes: [`frontend/README.md`](frontend/README.md)
- Russian README: [`README_RU.md`](README_RU.md)

## Quick Start

Install Python dependencies:

```bash
poetry install --with dev
```

Create a local environment file and replace placeholders. Do not commit
`.env.dev`.

```bash
cp .env.example .env.dev
```

Start the external OpenAI-compatible inference server, then validate and run the
dev stack:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml config
docker compose --env-file .env.dev -f docker-compose-dev.yml up --build
```

For container code sync during development:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml watch backend rag-worker frontend
```

Common local URLs:

- Frontend: `http://127.0.0.1:5173`
- Backend API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/v1/health/ready`
- Backend metrics: `http://127.0.0.1:8000/metrics`
- Worker metrics: `http://127.0.0.1:9100/metrics`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

## CLI

Project commands are exposed through [`commands.py`](commands.py):

```bash
poetry run python commands.py --help
poetry run python commands.py <subcommand> --help
```

Available subcommands:

- `parse-pdf`: parse clinical guideline PDFs into sections JSONL.
- `generate-qa`: generate a synthetic QA dataset from parsed sections.
- `metrics-eval-full`: run answer generation, RAGAS, and retrieval metrics.
- `metrics-eval-retriever`: run retrieve + rerank evaluation without answer
  generation.

See [`docs/commands.md`](docs/commands.md) for options, examples, and artifact
formats.

There is no interactive user-facing RAG CLI. User requests go through the
FastAPI chat API and are processed asynchronously by `rag-worker`.

## Docker Compose

Use the dev compose file from the repository root:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up --build
```

The stack includes:

- `postgres`, `redis`, `qdrant`
- `dvc-pull` for downloading `data.dvc` into the shared `rag-corpus` volume
- `migrations` for `alembic upgrade head`
- `backend` for `uvicorn cadence_md.backend.main:app`
- `rag-worker` for the Celery RAG queue
- `frontend` for the Vite dev server
- `prometheus`, `grafana`

The inference server is intentionally external to `docker-compose-dev.yml`.
Configure it with `MODEL_INFERENCE_BASE_URL` and `MODEL_INFERENCE_API_KEY`.

## Frontend

Run the frontend locally:

```bash
npm --prefix frontend install
VITE_API_BASE_URL= VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm --prefix frontend run dev
```

The app talks to `/api/v1`. Leaving `VITE_API_BASE_URL` empty enables the Vite
same-origin proxy and avoids local CORS requirements. Auth stores only the
short-lived access token in `sessionStorage`; logout and API `401` responses
clear client state and require re-login.

## Quality Checks

Backend unit tests:

```bash
poetry run pytest -m "not integration"
```

Backend lint and format checks:

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

Integration dependencies are defined in `docker-compose-test.yml`:

```bash
docker compose -f docker-compose-test.yml up -d postgres-test redis-test
INTEGRATION_DATABASE_URL=postgresql+asyncpg://cadence_md:cadence_md_dev@127.0.0.1:55432/cadence_md_test \
INTEGRATION_REDIS_URL=redis://127.0.0.1:56379/0 \
poetry run pytest -m integration
```

## License

Proprietary - see [`LICENSE.md`](LICENSE.md).
