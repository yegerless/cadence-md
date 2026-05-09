# CADENCE-MD Deployment

This guide describes production deployment for CADENCE-MD. The current
production path is Docker Compose. Kubernetes deployment is intentionally left
as a future section.

Inference can run in two modes: an external OpenAI-compatible service, or the
optional Docker Compose `inference` profile with vLLM containers on a Linux
server with NVIDIA GPU support.

## Production Architecture

The production compose stack exposes only the edge nginx service by default.
Stateful services and application containers stay on Docker networks.

- `nginx`: public reverse proxy and load balancer.
- `frontend`: nginx container serving the built React SPA.
- `backend`: FastAPI application behind nginx.
- `rag-worker`: Celery worker processing asynchronous RAG requests.
- `postgres`, `redis`, `qdrant`: internal data services.
- `dvc-pull`: one-shot corpus downloader into the `rag-corpus` volume.
- `migrations`: one-shot `alembic upgrade head`.
- `prometheus`, `grafana`: observability services bound to localhost by default.

The browser talks to a single origin. Static assets are served by frontend
nginx, and `/api/*` is proxied by edge nginx to FastAPI. This avoids CORS and
keeps the frontend production build independent from a hard-coded API host.

## Prerequisites

- Linux server with Docker Engine and Docker Compose.
- Git access to the repository.
- DNS record pointing your production hostname to the server or to an external
  load balancer.
- Firewall that allows public HTTP/HTTPS only as needed.
- DVC credentials for the Yandex Cloud S3-compatible corpus remote.
- A reachable external OpenAI-compatible inference server, or a Linux server
  prepared for the optional vLLM inference profile.
- Production secrets for JWT, Postgres, Qdrant, inference, S3, and Grafana.

Do not store production secrets in git. Keep them in `.env.prod` on the server
or in your secret manager.

## Docker Compose Deploy

### 1. Prepare The Server

Install Docker and confirm Compose is available:

```bash
docker --version
docker compose version
```

Clone the repository:

```bash
git clone git@github.com:yegerless/cadence-md.git
cd cadence-md
```

Create the production environment file:

```bash
cp .env.prod.example .env.prod
```

Edit `.env.prod` and replace every `replace-with-...` value. The most important
settings are:

- `POSTGRES_PASSWORD` and matching password inside `DATABASE_URL`.
- `JWT_SECRET`, at least 32 random characters.
- `QDRANT__SERVICE__API_KEY`.
- `MODEL_INFERENCE_BASE_URL` and `MODEL_INFERENCE_API_KEY`.
- `S3_KEY_ID` and `S3_KEY`.
- `GRAFANA_ADMIN_PASSWORD`.
- `CADENCE_PUBLIC_HOST` and `NGINX_HTTP_PORT`.

Keep `VITE_API_BASE_URL=` explicitly empty unless you intentionally serve the
API from a different origin. The production frontend build uses this empty value
to send requests to `/api/v1` through edge nginx.

### 2. Configure Inference

CADENCE-MD expects one OpenAI-compatible base URL in `MODEL_INFERENCE_BASE_URL`.
The application uses that URL for embeddings, reranking, chat completions, and
the readiness `/models` check.

#### External Inference Server

Use this mode when inference runs outside the production compose stack. The
server must be reachable from Docker containers and expose:

- `/v1/models` with `bge-m3`, `bge-reranker-v2-m3`, and `medgemma`.
- `/v1/embeddings` for `bge-m3`.
- `/v1/rerank` for `bge-reranker-v2-m3`.
- `/v1/chat/completions` for `medgemma`.

Configure `.env.prod` with a real network URL:

```bash
MODEL_INFERENCE_BASE_URL=https://inference.example.com/v1
MODEL_INFERENCE_API_KEY=<your_inference_api_key>
```

If the inference service runs on the same Docker host, make sure the URL is
reachable from containers. A host-local setup may need a compose override that
adds `host.docker.internal:host-gateway` to `backend` and `rag-worker`.

#### Optional vLLM Inference Profile

Use this mode when the production server is a Linux host with NVIDIA GPU support
and you want Compose to start inference together with the application. Install
and validate the NVIDIA Container Toolkit first:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu24.04 nvidia-smi
```

The `inference` profile starts four internal services:

- `vllm-embeddings`: `bge-m3`, pooling/embed runner, context `4096`.
- `vllm-reranker`: `bge-reranker-v2-m3`, pooling/rerank service, context `4096`.
- `vllm-llm`: `medgemma`, generation runner, context `24576`.
- `inference-gateway`: internal nginx gateway that exposes one
  `http://inference-gateway:8080/v1` base URL to `backend` and `rag-worker`.

Update `.env.prod` for this mode:

```bash
MODEL_INFERENCE_BASE_URL=http://inference-gateway:8080/v1
MODEL_INFERENCE_API_KEY=<your_internal_inference_api_key>
HF_TOKEN=<optional_hugging_face_token>
```

Review the `VLLM_*` variables in `.env.prod.example`. The served model aliases
match `llama-configs/models-test.ini` and the app defaults. The actual Hugging
Face or GGUF checkpoints may need model-specific `VLLM_*_EXTRA_ARGS`, for
example `--load-format gguf`, `--tensor-parallel-size`, or `--pooler-config`.

### 3. Validate Configuration

Render the compose configuration before starting services:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml config
```

Validate nginx configuration syntax:

```bash
docker run --rm \
  -v "$PWD/infra/nginx/prod.conf:/etc/nginx/conf.d/default.conf:ro" \
  nginx:stable-alpine nginx -t

docker run --rm \
  -v "$PWD/infra/nginx/frontend.conf:/etc/nginx/conf.d/default.conf:ro" \
  nginx:stable-alpine nginx -t

docker run --rm \
  --add-host vllm-embeddings:127.0.0.1 \
  --add-host vllm-reranker:127.0.0.1 \
  --add-host vllm-llm:127.0.0.1 \
  -v "$PWD/infra/nginx/inference-gateway.conf:/etc/nginx/conf.d/default.conf:ro" \
  nginx:stable-alpine nginx -t
```

Validate the optional vLLM profile configuration without starting GPU
containers:

```bash
docker compose --profile inference --env-file .env.prod -f docker-compose-prod.yml config
```

### 4. Build And Start

Build images and start the stack:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --build
```

When using the optional vLLM profile, include the profile flag:

```bash
docker compose --profile inference --env-file .env.prod -f docker-compose-prod.yml up -d --build
```

The first startup runs `dvc-pull` and `migrations` before `backend` and
`rag-worker`. The corpus is stored in the named `rag-corpus` volume and mounted
read-only by application containers.

Check service status:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml ps
```

Follow logs:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml logs -f nginx backend rag-worker
```

For the vLLM profile, include inference services:

```bash
docker compose --profile inference --env-file .env.prod -f docker-compose-prod.yml \
  logs -f inference-gateway vllm-embeddings vllm-reranker vllm-llm backend rag-worker
```

### 5. Health Checks

Check edge nginx:

```bash
curl -fsS "http://${CADENCE_PUBLIC_HOST:-localhost}/nginx-health"
```

Check backend readiness through nginx:

```bash
curl -fsS "http://${CADENCE_PUBLIC_HOST:-localhost}/api/v1/health/ready"
```

Check the frontend:

```bash
curl -fsS "http://${CADENCE_PUBLIC_HOST:-localhost}/" >/dev/null
```

With the vLLM profile, check the internal gateway from the compose network:

```bash
docker compose --profile inference --env-file .env.prod -f docker-compose-prod.yml \
  run --rm --no-deps backend python -c \
  "import urllib.request; print(urllib.request.urlopen('http://inference-gateway:8080/v1/models', timeout=5).read().decode())"
```

If readiness returns `503`, inspect backend logs and the JSON response. Common
causes are unavailable inference API, missing Qdrant collection schema, failed
Postgres connectivity, or Redis connectivity issues. For the vLLM profile, also
check GPU memory, Hugging Face download/authentication errors, model checkpoint
compatibility, `/v1/rerank` support, and aliases that no longer match backend
settings.

### 6. Scaling

Scale stateless backend workers with Compose:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --scale backend=2
```

Edge nginx balances requests through the `backend:8000` Docker service name.
After changing the number of backend replicas, restart nginx so it refreshes
Docker DNS resolution:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml restart nginx
```

Scale `rag-worker` separately when queue throughput is the bottleneck:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --scale rag-worker=2
```

Do not scale `postgres`, `redis`, or the single-node `qdrant` service with this
compose file.

### 7. Updates And Rollback

Deploy a new revision:

```bash
git pull --ff-only
docker compose --env-file .env.prod -f docker-compose-prod.yml config
docker compose --env-file .env.prod -f docker-compose-prod.yml build backend rag-worker frontend
docker compose --env-file .env.prod -f docker-compose-prod.yml up --no-deps --force-recreate migrations
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --remove-orphans
```

If the DVC corpus changed, refresh it explicitly:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml up --no-deps --force-recreate dvc-pull
docker compose --env-file .env.prod -f docker-compose-prod.yml restart backend rag-worker
```

Rollback to the previous git revision or image tag, then rebuild/recreate the
application services:

```bash
git checkout <previous-revision>
docker compose --env-file .env.prod -f docker-compose-prod.yml build backend rag-worker frontend
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --force-recreate backend rag-worker frontend nginx
```

Database migrations are not automatically reversible. Review Alembic downgrade
steps before rolling back across schema changes.

### 8. Backups

Back up Postgres regularly:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml exec -T postgres \
  sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' > cadence_md_postgres.sql
```

Back up named Docker volumes from the host or with your infrastructure backup
tool:

- `postgres-data`: database files.
- `redis-data`: Redis append-only data.
- `qdrant`: vector storage.
- `qdrant-snapshots`: Qdrant snapshots.
- `grafana-data`: dashboards and Grafana state.

Qdrant snapshots are configured under `/qdrant/snapshots` and persisted in the
`qdrant-snapshots` volume. Trigger snapshots through the Qdrant API from a
trusted internal network client and keep the API key out of shell history.

The `rag-corpus` volume can be recreated from DVC with `dvc-pull`, so treat it
as reproducible cache unless your operational policy requires volume-level
backups.

### 9. Security Checklist

- Publish only nginx to the public network.
- Keep Postgres, Redis, Qdrant, backend, frontend, Prometheus, and Grafana ports
  private.
- Bind Prometheus and Grafana to `127.0.0.1` unless they are behind VPN or a
  protected admin proxy.
- Use strong, unique values for all required secrets in `.env.prod`.
- Terminate TLS at an external load balancer, reverse proxy, or a future nginx
  TLS override. Do not send production credentials over plain HTTP.
- Keep `LANGFUSE_TRACE_QUERY_MODE=redacted` unless full medical text tracing is
  explicitly approved.
- Restrict SSH and Docker access to trusted operators only.

## Kubernetes Deploy

Planned. This section will describe manifests, secrets, ingress, persistent
volumes, jobs for migrations and DVC corpus loading, and horizontal scaling.
