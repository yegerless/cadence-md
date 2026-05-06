# Project Setup Instruction

Follow the instructions below to setup the project.

### Installation Steps

#### 1. Clone the Repository

Clone the project repository from GitHub to your local machine:

```bash
git clone git@github.com:CORESIGHT-Health/simple-randomiser.git
cd simple-randomiser
```

#### 2. Init virtual environment and install dependencies

Run next command:

```
poetry install
```

If you want to install all dependencies (include development) use next command:

```
poetry install --with dev
```

#### 3. Install pre-commit hook

Pre-commit tool configured yet, you need only install git hook:

```bash
poetry run pre-commit install
```

#### 4. Get data from DVC remote storage

In that project we use Yandex Cloud object storage as dvc remote storage to
store texts corpus of russian clinical recommendations. The basic settings for
connecting to the remote storage of the DVC are already written in the file
.dvc/config. Use next commands to set up DVC remote storage security settings
and download texts corpus to your local computer.

```bash
# Add credentials for connection to remote DVC storage or use .dvc/config.local.example
poetry run dvc remote modify --local yandex access_key_id '<your_access_key_id>'
poetry run dvc remote modify --local yandex secret_access_key '<your_secret_access_key>'

# Download data from DVC remote storage
poetry run dvc pull
```

When using `docker-compose-dev.yml`, the `dvc-pull` service downloads `data.dvc`
into the shared `rag-corpus` volume automatically. The same volume is mounted
read-only into `rag-worker` for optional Qdrant indexing and into `backend` for
authenticated source PDF downloads.

#### Additional info

Always run pre-commit before create new commits:

```bash
poetry run pre-commit run -av
```

All pre-commit checks should not be red.

Export dependencies from poetry in requirements.txt for running project in
docker:

```bash
poetry export --without-hashes --format=requirements.txt --without dev > requirements.txt
```

You can use next command to generate QA validation dataset (you need GigaChat
API Key):

```bash
python commands.py generate-qa

# Use for get help with command options
python commands.py generate-qa --help
```

#### 5. Run local dev infrastructure

Create a local env file from the committed template and replace placeholder
values as needed. Do not commit `.env.dev`.

```bash
cp .env.example .env.dev
```

Validate the compose file with the explicit env file:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml config
```

Start the full dev service stack. The OpenAI-compatible inference server is not
part of this compose file, so start it separately first and point
`MODEL_INFERENCE_BASE_URL` at it.

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up --build
```

This compose flow is the primary way to run RAG in development: backend accepts
chat requests and `rag-worker` processes them asynchronously. There is no
interactive user REPL entrypoint for RAG.

If you only want infrastructure dependencies, start them explicitly:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up postgres redis qdrant
```

Run database migrations manually when needed:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml run --rm migrations
```

The dev compose file also contains the FastAPI backend and the RAG Celery
worker. Starting `backend` or `rag-worker` through compose waits for successful
migrations:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up backend
```

Start the RAG worker when you want asynchronous chat requests to be processed:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up rag-worker
```

Prometheus and Grafana are available for local observability:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up prometheus grafana
```

Check service health and metrics:

```bash
curl -fsS http://127.0.0.1:8000/api/v1/health/live
curl -fsS http://127.0.0.1:8000/api/v1/health/ready
curl -fsS http://127.0.0.1:8000/api/v1/health/rag
docker compose --env-file .env.dev -f docker-compose-dev.yml exec rag-worker \
  python -m cadence_md.workers.rag_health --timeout 5
curl -fsS http://127.0.0.1:8000/metrics
curl -fsS http://127.0.0.1:9100/metrics
```

Backend metrics are exposed at `http://127.0.0.1:8000/metrics`; worker metrics
are exposed at `http://127.0.0.1:9100/metrics`. Prometheus is available at
`http://127.0.0.1:9090`; Grafana is available at `http://127.0.0.1:3000`.

The React SPA lives in `frontend/` and can be started with the compose frontend
profile:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml --profile frontend up --build frontend
```

By default `VITE_API_BASE_URL` is empty and the Vite dev server proxies
`/api/v1/...` to `VITE_API_PROXY_TARGET` (`http://backend:8000` in compose).
This keeps local browser traffic same-origin and avoids requiring CORS
middleware in the backend. For a host-only run use:

```bash
cd frontend
npm install
VITE_API_BASE_URL= VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
npm run build
npm test
```

Frontend auth stores only the short-lived access token in `sessionStorage`; no
refresh token is used in the MVP. Logout and any API `401` clear client auth
state and redirect to `/login`. Smoke-check logout, automatic `401` redirect,
chat submit/polling, cancel, retry, and source rendering after backend changes.

#### 5.1 Test stack and test commands

Bring up dedicated integration dependencies (tests do not auto-start services):

```bash
docker compose -f docker-compose-test.yml up -d postgres-test redis-test
```

Optional Qdrant smoke target:

```bash
docker compose -f docker-compose-test.yml --profile with-qdrant up -d qdrant-test
```

Compose smoke validation:

```bash
docker compose -f docker-compose-test.yml config
docker compose -f docker-compose-dev.yml config
```

Run fast unit tests only:

```bash
poetry run pytest -m "not integration"
```

Run integration and smoke tests:

```bash
INTEGRATION_DATABASE_URL=postgresql+asyncpg://cadence_md:cadence_md_dev@127.0.0.1:55432/cadence_md_test \
INTEGRATION_REDIS_URL=redis://127.0.0.1:56379/0 \
poetry run pytest -m integration
```

Frontend smoke and build:

```bash
cd frontend
npm test
npm run build
```

Langfuse is disabled by default and is treated as an external service for this
dev compose file. If you enable `LANGFUSE_ENABLED=true`, keep real
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` only in `.env.dev`. Full medical
query text is redacted unless `LANGFUSE_TRACE_QUERY_MODE=full` is explicitly
configured.

The inference server is external to `docker-compose-dev.yml` and must be
available at `MODEL_INFERENCE_BASE_URL`.

#### 6. Run llama.cpp inference server (embedder, reranker, llm)

If you use Mac llama.cpp is single opportunity to run reranker models with
OpenAI-compatible API. On linux use vLLM.

The project uses one OpenAI-compatible base URL (`MODEL_INFERENCE_BASE_URL`) for
models inference. To run project you need:

- embedder;
- reranker;
- LLM for text generation.

Model names are taken from `llama-configs/*.ini`. You may use an existing file
or create your own.

Install llama.cpp once:

```bash
brew install llama.cpp
```

You can use llama-cli for download GGUF files from Hugging Face:

```bash
llama-cli --hf-repo repo-owner/hf-repo --hf-file filename.gguf
```

By default (on Mac), downloaded files are stored in local HF cache:
`~/.cache/huggingface/hub`

Examples of `llama-server` startup commands:

```bash
llama-server \
  --models-dir /path/to/you/models \
  --models-preset /path/to/you/models.ini \
  --models-max 3 \
  --metrics \
  --perf \
  --log-timestamps \
  --log-prefix
```

Important:

- `cadence_md` currently uses one `MODEL_INFERENCE_BASE_URL` for all model
  types;
