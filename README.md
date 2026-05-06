# CADENCE-MD 🏥

**C**linical **A**ssistant for **D**iagnosis, **E**vidence **N**avigation &
**C**ase **E**valuation for **M**edical **D**octors

> AI-powered clinical decision support system for physicians

## Overview

CADENCE-MD is an intelligent medical assistant that helps doctors make
evidence-based decisions by providing instant access to clinical guidelines and
the latest medical research through advanced RAG architecture and PubMed
integration.

## Key Features

- **Clinical Guidelines RAG**: Semantic search through medical protocols and
  guidelines
- **PubMed Integration**: Real-time access to latest medical research
- **Diagnostic Support**: AI-assisted differential diagnosis and treatment
  recommendations
- **Physician-Centric**: Designed specifically for clinical workflow integration

## Technology Stack

- LLM + RAG Architecture
- Vector Databases & Semantic Search
- PubMed API Integration
- Medical NLP Processing

## Setup project

You can find project setup instruction in docs/setup.md

## CLI commands

Project tasks are exposed via [`commands.py`](commands.py). Run from the
repository root with Poetry:

```bash
poetry run python commands.py --help
poetry run python commands.py <subcommand> --help
```

The user-facing interactive RAG CLI has been removed. Run user requests through
the FastAPI backend (`/api/v1/chat/...`) with the `rag-worker` service, or use
the metrics pipeline commands below for offline evaluation.

### `parse-pdf`

Parses a directory of clinical guideline PDFs into a JSONL with extracted
sections (`ClinicalSection` records).

| Option          | Default | Description                            |
| --------------- | ------- | -------------------------------------- |
| `--pdf-dir`     | —       | Input directory containing `*.pdf`     |
| `--output-file` | —       | Output JSONL file with parsed sections |
| `--max-files`   | —       | Optional cap on number of input PDFs   |

Examples:

```bash
poetry run python commands.py parse-pdf \
  --pdf-dir data/main_specialities \
  --output-file data/clinical_sections.jsonl
```

```bash
poetry run python commands.py parse-pdf \
  --pdf-dir data/main_specialities \
  --output-file data/clinical_sections_sample.jsonl \
  --max-files 5
```

### `generate-qa`

Builds a synthetic QA dataset from a JSONL of clinical sections (parser output).

| Option               | Default                                             | Description                                                  |
| -------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| `--sections-file`    | `data/clinical_sections.jsonl`                      | Input sections JSONL                                         |
| `--output-file`      | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | Output QA JSONL                                              |
| `--model`            | `GigaChat-2-Max`                                    | LLM name (e.g. `GigaChat`, `GigaChat-2-Max`, `GigaChat-pro`) |
| `--temperature`      | `0.0`                                               | Sampling temperature                                         |
| `--max-context`      | `10000`                                             | Max context length (characters)                              |
| `--sections-per-pdf` | `3`                                                 | Randomly sample up to N sections from each source PDF        |
| `--seed`             | —                                                   | Random seed for reproducible section sampling                |

Example:

```bash
poetry run python commands.py generate-qa --sections-file data/clinical_sections.jsonl
```

On success, a Markdown generation report is written next to `--output-file`:
`{stem}_generation_report.md`, where `{stem}` is the output basename without
extension (for example `.../qa_dataset.jsonl` →
`.../qa_dataset_generation_report.md`). The report lists CLI parameters, a
pipeline summary (sections loaded, invalid JSONL lines skipped, pairs generated,
failed sections), and counts by question type and section type.

Requires LLM credentials (e.g. `GIGACHAT_API_KEY`) as configured for the QA
generator. The command fails if `--output-file` already exists to avoid
accidental appends to stale datasets.

### `metrics-eval-full`

Full RAG evaluation: answer generation, RAGAS metrics, retrieval metrics, and
reports under the output directory.

| Option                          | Default                                             | Description                                    |
| ------------------------------- | --------------------------------------------------- | ---------------------------------------------- |
| `--dataset-file`                | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | Evaluation QA JSONL                            |
| `--output-dir`                  | `metrics/results/`                                  | Reports and metric files                       |
| `--sample-size`                 | —                                                   | Limit the number of test cases                 |
| `--k`                           | `5`                                                 | K for recall@K / precision@K                   |
| `--enable-text-matcher-metrics` | off                                                 | Also compute diagnostic `text_match_*` metrics |

Example:

```bash
poetry run python commands.py metrics-eval-full --sample-size 20
```

Expects a running Qdrant instance, an indexed corpus, and the rest of the stack
described in the setup guide.

### `metrics-eval-retriever`

Retriever-only evaluation (retrieve + rerank): no RAGAS and no full answer
generation.

Same options as `metrics-eval-full`, plus:

| Option      | Default          | Description                                                     |
| ----------- | ---------------- | --------------------------------------------------------------- |
| `--k`       | pipeline default | K for recall@K / precision@K                                    |
| `--workers` | `1`              | Parallel worker threads for independent retrieve + rerank cases |

Both `full` and `retriever` modes now create a dedicated run directory under
`--output-dir`:

```text
metrics/results/<run_id>/
  run_manifest.json
  summary_metrics.json
  report.md
  errors.jsonl
  # full mode
  cases.jsonl
  ragas_scores.parquet
  # retriever mode
  retrieval_cases.jsonl
```

Primary retrieval metrics are section-based and require `section_id` in both the
QA dataset and indexed Qdrant metadata. Regenerate the QA dataset and reindex
the corpus after this change. Use `--enable-text-matcher-metrics` only for
optional diagnostic text-overlap metrics; they are written separately as
`text_match_*`.

Example:

```bash
poetry run python commands.py metrics-eval-retriever --k 5
```

See `poetry run python commands.py metrics-eval-full --help` and
`metrics-eval-retriever --help` for all evaluation flags.

## Service Health

The FastAPI backend exposes versioned health probes for local Docker Compose and
future orchestration:

- `GET /api/v1/health/live` checks only that the backend process is alive.
- `GET /api/v1/health/ready` checks Postgres, Redis, Qdrant collection/schema,
  and the OpenAI-compatible inference API. It returns `503` when a required
  dependency is unavailable.
- `GET /api/v1/health/rag` returns detailed RAG dependency status for Qdrant,
  inference, and the queue broker without exposing API keys or secrets.

Graceful degradation policy: backend startup does not construct the heavy RAG
stack and does not fail just because Qdrant or inference is temporarily
unavailable. Readiness reports `503` in that state so Docker/orchestrators can
stop routing traffic. The chat API remains asynchronous:
`POST /api/v1/chat/messages` can still persist and enqueue a request when
Postgres, Redis, and Celery are available; the worker records a controlled
failure if RAG execution later cannot reach Qdrant or inference.

## Frontend SPA

The React frontend is in `frontend/` and talks to the versioned backend API
under `/api/v1`. Start it locally with:

```bash
cd frontend
npm install
VITE_API_BASE_URL= VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
npm run build
npm test
```

Or run it through Docker Compose:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml --profile frontend up --build frontend
```

`VITE_API_BASE_URL` may be left empty for the Vite same-origin proxy. Auth
stores only the short-lived access token in `sessionStorage`; logout and API
`401` responses clear client state and require re-login.

## Test Commands

Use dedicated integration dependencies from `docker-compose-test.yml`:

```bash
docker compose -f docker-compose-test.yml up -d postgres-test redis-test
docker compose -f docker-compose-test.yml config
docker compose -f docker-compose-dev.yml config
```

Run fast unit tests:

```bash
poetry run pytest -m "not integration"
```

Run backend/worker integration and smoke tests:

```bash
INTEGRATION_DATABASE_URL=postgresql+asyncpg://cadence_md:cadence_md_dev@127.0.0.1:55432/cadence_md_test \
INTEGRATION_REDIS_URL=redis://127.0.0.1:56379/0 \
poetry run pytest -m integration
```

## Observability

The backend exposes Prometheus metrics at `GET /metrics` outside the versioned
user API. The RAG worker exposes its own metrics endpoint on
`WORKER_METRICS_PORT` in Docker Compose, and Prometheus/Grafana dev services are
configured in `docker-compose-dev.yml`.

Start the local observability stack with the service stack:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up --build
```

Grafana is available on `http://127.0.0.1:3000` by default, Prometheus on
`http://127.0.0.1:9090`, backend metrics on `http://127.0.0.1:8000/metrics`, and
worker metrics on `http://127.0.0.1:9100/metrics`. See `docs/setup.md` for
health checks, migrations, and partial service startup commands. Langfuse
tracing is disabled by default; enable it with `LANGFUSE_ENABLED=true` and real
`LANGFUSE_*` credentials in `.env.dev`. Medical queries are redacted from logs
and Langfuse by default; use `LANGFUSE_TRACE_QUERY_MODE=full` only for
explicitly approved debugging.

## License

Proprietary - See [LICENSE.md](LICENSE.md) for details.

---

**Disclaimer**: This is a research project. Not for direct clinical use without
validation.
