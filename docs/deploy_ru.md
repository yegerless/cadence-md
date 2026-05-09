# Деплой CADENCE-MD

Эта инструкция описывает production-деплой CADENCE-MD. В репозитории есть два
production-пути: Docker Compose для одного сервера и Kubernetes manifests в
`infra/k8s/` для деплоя в кластер.

Inference можно запускать в двух режимах: внешний OpenAI-compatible service или
опциональный Docker Compose profile `inference` с vLLM containers на Linux
сервере с NVIDIA GPU support.

## Production Architecture

Production compose stack по умолчанию публикует наружу только edge nginx.
Stateful services и application containers остаются внутри Docker networks.

- `nginx`: публичный reverse proxy и балансировщик нагрузки.
- `frontend`: nginx container, который раздаёт собранный React SPA.
- `backend`: FastAPI-приложение за nginx.
- `rag-worker`: Celery worker для асинхронной обработки RAG-запросов.
- `postgres`, `redis`, `qdrant`: внутренние data services.
- `dvc-pull`: one-shot загрузка корпуса в volume `rag-corpus`.
- `migrations`: one-shot `alembic upgrade head`.
- `prometheus`, `grafana`: observability services, по умолчанию привязаны к
  localhost.

Браузер работает с одним origin. Static assets раздаёт frontend nginx, а
`/api/*` edge nginx проксирует в FastAPI. Так не нужен CORS, а production build
frontend не получает захардкоженный API host.

## Требования

- Linux-сервер с Docker Engine и Docker Compose.
- Git-доступ к репозиторию.
- DNS-запись, которая указывает production hostname на сервер или внешний load
  balancer.
- Firewall, который открывает публично только HTTP/HTTPS по необходимости.
- DVC credentials для Yandex Cloud S3-compatible corpus remote.
- Доступный внешний OpenAI-compatible inference server или Linux-сервер,
  подготовленный для optional vLLM inference profile.
- Production secrets для JWT, Postgres, Qdrant, inference, S3 и Grafana.

Не храните production secrets в git. Держите их в `.env.prod` на сервере или в
вашем secret manager.

## Docker Compose Deploy

### 1. Подготовка сервера

Установите Docker и проверьте, что Compose доступен:

```bash
docker --version
docker compose version
```

Склонируйте репозиторий:

```bash
git clone git@github.com:yegerless/cadence-md.git
cd cadence-md
```

Создайте production env-файл:

```bash
cp .env.prod.example .env.prod
```

Отредактируйте `.env.prod` и замените все значения `replace-with-...`. Самые
важные настройки:

- `POSTGRES_PASSWORD` и такой же пароль внутри `DATABASE_URL`.
- `JWT_SECRET`, минимум 32 случайных символа.
- `QDRANT__SERVICE__API_KEY`.
- `MODEL_INFERENCE_BASE_URL` и `MODEL_INFERENCE_API_KEY`.
- `S3_KEY_ID` и `S3_KEY`.
- `GRAFANA_ADMIN_PASSWORD`.
- `CADENCE_PUBLIC_HOST` и `NGINX_HTTP_PORT`.

Оставьте `VITE_API_BASE_URL=` явно пустым, если вы не выносите API на другой
origin. Production frontend build использует это пустое значение, чтобы ходить в
`/api/v1` через edge nginx.

### 2. Настройка inference

CADENCE-MD ожидает один OpenAI-compatible base URL в `MODEL_INFERENCE_BASE_URL`.
Приложение использует этот URL для embeddings, reranking, chat completions и
readiness-проверки `/models`.

#### Внешний inference server

Используйте этот режим, если inference запускается вне production compose stack.
Сервер должен быть доступен из Docker containers и отдавать:

- `/v1/models` с `bge-m3`, `bge-reranker-v2-m3` и `medgemma`.
- `/v1/embeddings` для `bge-m3`.
- `/v1/rerank` для `bge-reranker-v2-m3`.
- `/v1/chat/completions` для `medgemma`.

Укажите в `.env.prod` реальный сетевой URL:

```bash
MODEL_INFERENCE_BASE_URL=https://inference.example.com/v1
MODEL_INFERENCE_API_KEY=<your_inference_api_key>
```

Если inference service работает на том же Docker host, убедитесь, что URL
доступен из контейнеров. Host-local setup может потребовать compose override с
`host.docker.internal:host-gateway` для `backend` и `rag-worker`.

#### Опциональный vLLM inference profile

Используйте этот режим, если production server — Linux host с NVIDIA GPU и вы
хотите запускать inference вместе с приложением через Compose. Сначала
установите и проверьте NVIDIA Container Toolkit:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu24.04 nvidia-smi
```

Profile `inference` запускает четыре внутренних сервиса:

- `vllm-embeddings`: `bge-m3`, pooling/embed runner, context `4096`.
- `vllm-reranker`: `bge-reranker-v2-m3`, pooling/rerank service, context `4096`.
- `vllm-llm`: `medgemma`, generation runner, context `24576`.
- `inference-gateway`: внутренний nginx gateway, который даёт `backend` и
  `rag-worker` один base URL `http://inference-gateway:8080/v1`.

Для этого режима обновите `.env.prod`:

```bash
MODEL_INFERENCE_BASE_URL=http://inference-gateway:8080/v1
MODEL_INFERENCE_API_KEY=<your_internal_inference_api_key>
HF_TOKEN=<optional_hugging_face_token>
```

Проверьте переменные `VLLM_*` в `.env.prod.example`. Served model aliases
совпадают с `llama-configs/models-test.ini` и app defaults. Конкретные Hugging
Face или GGUF checkpoints могут требовать model-specific `VLLM_*_EXTRA_ARGS`,
например `--load-format gguf`, `--tensor-parallel-size` или `--pooler-config`.

### 3. Проверка конфигурации

Перед стартом сервисов отрендерите compose configuration:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml config
```

Проверьте синтаксис nginx configs:

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

Проверьте optional vLLM profile configuration без запуска GPU containers:

```bash
docker compose --profile inference --env-file .env.prod -f docker-compose-prod.yml config
```

### 4. Build и старт

Соберите образы и запустите stack:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --build
```

Если используете optional vLLM profile, добавьте profile flag:

```bash
docker compose --profile inference --env-file .env.prod -f docker-compose-prod.yml up -d --build
```

Первый старт запускает `dvc-pull` и `migrations` до `backend` и `rag-worker`.
Корпус хранится в named volume `rag-corpus` и монтируется application containers
в read-only режиме.

Проверьте статус сервисов:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml ps
```

Смотрите логи:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml logs -f nginx backend rag-worker
```

Для vLLM profile добавьте inference services:

```bash
docker compose --profile inference --env-file .env.prod -f docker-compose-prod.yml \
  logs -f inference-gateway vllm-embeddings vllm-reranker vllm-llm backend rag-worker
```

### 5. Health checks

Проверьте edge nginx:

```bash
curl -fsS "http://${CADENCE_PUBLIC_HOST:-localhost}/nginx-health"
```

Проверьте backend readiness через nginx:

```bash
curl -fsS "http://${CADENCE_PUBLIC_HOST:-localhost}/api/v1/health/ready"
```

Проверьте frontend:

```bash
curl -fsS "http://${CADENCE_PUBLIC_HOST:-localhost}/" >/dev/null
```

С vLLM profile проверьте internal gateway из compose network:

```bash
docker compose --profile inference --env-file .env.prod -f docker-compose-prod.yml \
  run --rm --no-deps backend python -c \
  "import urllib.request; print(urllib.request.urlopen('http://inference-gateway:8080/v1/models', timeout=5).read().decode())"
```

Если readiness возвращает `503`, посмотрите backend logs и JSON response. Частые
причины: недоступный inference API, отсутствующая Qdrant collection schema,
проблемы соединения с Postgres или Redis. Для vLLM profile также проверьте GPU
memory, ошибки Hugging Face download/authentication, совместимость model
checkpoint, поддержку `/v1/rerank` и aliases, которые должны совпадать с backend
settings.

### 6. Масштабирование

Масштабируйте stateless backend containers через Compose:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --scale backend=2
```

Edge nginx балансирует запросы через Docker service name `backend:8000`. После
изменения числа backend replicas перезапустите nginx, чтобы он перечитал Docker
DNS:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml restart nginx
```

Масштабируйте `rag-worker` отдельно, если узкое место — очередь:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --scale rag-worker=2
```

Не масштабируйте `postgres`, `redis` и single-node `qdrant` этим compose-файлом.

### 7. Обновления и rollback

Задеплойте новую ревизию:

```bash
git pull --ff-only
docker compose --env-file .env.prod -f docker-compose-prod.yml config
docker compose --env-file .env.prod -f docker-compose-prod.yml build backend rag-worker frontend
docker compose --env-file .env.prod -f docker-compose-prod.yml up --no-deps --force-recreate migrations
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --remove-orphans
```

Если изменился DVC corpus, обновите его явно:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml up --no-deps --force-recreate dvc-pull
docker compose --env-file .env.prod -f docker-compose-prod.yml restart backend rag-worker
```

Rollback: вернитесь на предыдущую git revision или image tag, затем пересоберите
и пересоздайте application services:

```bash
git checkout <previous-revision>
docker compose --env-file .env.prod -f docker-compose-prod.yml build backend rag-worker frontend
docker compose --env-file .env.prod -f docker-compose-prod.yml up -d --force-recreate backend rag-worker frontend nginx
```

Database migrations не всегда обратимы автоматически. Перед rollback через
изменения схемы отдельно проверьте Alembic downgrade steps.

### 8. Backups

Регулярно делайте backup Postgres:

```bash
docker compose --env-file .env.prod -f docker-compose-prod.yml exec -T postgres \
  sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' > cadence_md_postgres.sql
```

Делайте backup named Docker volumes средствами хоста или инфраструктурным backup
tool:

- `postgres-data`: файлы базы данных.
- `redis-data`: Redis append-only data.
- `qdrant`: vector storage.
- `qdrant-snapshots`: Qdrant snapshots.
- `grafana-data`: dashboards и состояние Grafana.

Qdrant snapshots настроены в `/qdrant/snapshots` и сохраняются в volume
`qdrant-snapshots`. Запускайте snapshot creation через Qdrant API из доверенного
internal network client и не оставляйте API key в shell history.

Volume `rag-corpus` можно восстановить через DVC с помощью `dvc-pull`, поэтому
считайте его воспроизводимым cache, если ваша эксплуатационная политика не
требует volume-level backups.

### 9. Security checklist

- Публикуйте во внешнюю сеть только nginx.
- Держите Postgres, Redis, Qdrant, backend, frontend, Prometheus и Grafana ports
  приватными.
- Оставляйте Prometheus и Grafana на `127.0.0.1`, если они не закрыты VPN или
  protected admin proxy.
- Используйте сильные уникальные значения для всех обязательных secrets в
  `.env.prod`.
- Терминируйте TLS на внешнем load balancer, reverse proxy или будущей nginx TLS
  override-конфигурации. Не передавайте production credentials по plain HTTP.
- Оставляйте `LANGFUSE_TRACE_QUERY_MODE=redacted`, пока отправка полного
  медицинского текста в tracing явно не согласована.
- Ограничьте SSH и Docker access только доверенными операторами.

## Kubernetes Deploy

Kubernetes manifests лежат в `infra/k8s/` и повторяют production Compose
topology: один публичный edge nginx, внутренние application/data services, Jobs
для `dvc-pull` и database migrations, а также optional in-cluster vLLM
inference.

### 1. Build и push образов

Соберите три application images и отправьте их в registry, доступный кластеру:

```bash
docker build -f infra/docker/backend.Dockerfile \
  -t <registry>/cadence-md-backend:<tag> .
docker build -f infra/docker/rag-worker.Dockerfile \
  -t <registry>/cadence-md-rag-worker:<tag> .
docker build -f infra/docker/frontend.prod.Dockerfile \
  --build-arg VITE_API_BASE_URL= \
  -t <registry>/cadence-md-frontend:<tag> .

docker push <registry>/cadence-md-backend:<tag>
docker push <registry>/cadence-md-rag-worker:<tag>
docker push <registry>/cadence-md-frontend:<tag>
```

Обновите `infra/k8s/overlays/prod/kustomization.yaml` под ваш registry/tag или
выполните `kustomize edit set image` из этой директории.

### 2. Подготовка кластера

В кластере нужны:

- Default `StorageClass` для PVC или отдельные patches с явным storage class.
- Ingress controller, если используете `overlays/prod/ingress.yaml`.
- DNS для production hostname.
- NVIDIA device plugin и GPU nodes только для `overlays/inference`.

Base manifests создают namespace `cadence-md`. Перед применением проверьте PVC
sizes в `infra/k8s/base/storage/` и volume templates в StatefulSet manifests.
Self-hosted Postgres, Redis и Qdrant — это Compose-equivalent baseline. Для
более надёжного production setup используйте managed Postgres/Redis и замените
`DATABASE_URL`, `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` в
Secret/ConfigMap вместо применения этих StatefulSets.

### 3. Создание Secrets

Создавайте production secrets вне git. Ожидаемые ключи описаны в
`infra/k8s/base/secrets/README.md`:

```bash
kubectl apply -f infra/k8s/base/namespace.yaml

kubectl -n cadence-md create secret generic cadence-md-secret \
  --from-literal=POSTGRES_PASSWORD='<postgres-password>' \
  --from-literal=DATABASE_URL='postgresql+asyncpg://cadence_md:<postgres-password>@postgres:5432/cadence_md' \
  --from-literal=JWT_SECRET='<random-secret-at-least-32-characters>' \
  --from-literal=QDRANT__SERVICE__API_KEY='<qdrant-api-key>' \
  --from-literal=MODEL_INFERENCE_API_KEY='<inference-api-key>' \
  --from-literal=S3_KEY_ID='<yandex-s3-access-key-id>' \
  --from-literal=S3_KEY='<yandex-s3-secret-access-key>' \
  --from-literal=GRAFANA_ADMIN_PASSWORD='<grafana-admin-password>' \
  --from-literal=FLOWER_BASIC_AUTH='admin:<flower-password>' \
  --from-literal=LANGFUSE_PUBLIC_KEY='' \
  --from-literal=LANGFUSE_SECRET_KEY='' \
  --from-literal=HF_TOKEN=''
```

Для внешнего inference оставьте
`MODEL_INFERENCE_BASE_URL=https://inference.example.com/v1` в
`infra/k8s/base/configmaps/app-config.yaml` или переопределите его в вашем
production overlay. Endpoint должен отдавать `/v1/models`, `/v1/embeddings`,
`/v1/rerank`, `/v1/chat/completions` с model ids `bge-m3`, `bge-reranker-v2-m3`
и `medgemma`.

### 4. Render и apply

Перед применением посмотрите rendered manifests:

```bash
kubectl kustomize infra/k8s/overlays/prod
kubectl apply --dry-run=client -k infra/k8s/overlays/prod
```

Сначала примените persistent services:

```bash
kubectl apply -k infra/k8s/overlays/prod
kubectl -n cadence-md rollout status statefulset/postgres
kubectl -n cadence-md rollout status statefulset/redis
kubectl -n cadence-md rollout status statefulset/qdrant
```

Base overlay включает Jobs `dvc-pull` и `migrations`. Дождитесь их завершения,
прежде чем ожидать readiness от backend и worker:

```bash
kubectl -n cadence-md wait --for=condition=complete job/dvc-pull --timeout=30m
kubectl -n cadence-md wait --for=condition=complete job/migrations --timeout=10m
kubectl -n cadence-md rollout status deployment/backend
kubectl -n cadence-md rollout status deployment/rag-worker
kubectl -n cadence-md rollout status deployment/frontend
kubectl -n cadence-md rollout status deployment/nginx
```

Если нужно заново запустить `dvc-pull` или `migrations`, сначала удалите
completed Job и снова примените overlay:

```bash
kubectl -n cadence-md delete job dvc-pull migrations
kubectl apply -k infra/k8s/overlays/prod
```

### 5. Опциональный in-cluster vLLM

Используйте `infra/k8s/overlays/inference`, если в кластере есть NVIDIA GPU
nodes и вы хотите запускать vLLM в Kubernetes. Overlay добавляет
`vllm-embeddings`, `vllm-reranker`, `vllm-llm`, `inference-gateway`, shared
`vllm-cache` PVC и patch для `MODEL_INFERENCE_BASE_URL`:
`http://inference-gateway:8080/v1`.

Перед применением:

- Установите NVIDIA device plugin.
- Настройте `nodeSelector`, tolerations, GPU counts и `VLLM_*` values в overlay.
- Убедитесь, что access mode `vllm-cache` PVC поддерживается вашим storage
  backend. Если `ReadWriteMany` недоступен, используйте отдельные cache PVC или
  закрепите workloads на одном node со storage, который поддерживает выбранный
  режим.
- Укажите `HF_TOKEN` в `cadence-md-secret`, если checkpoints требуют доступ к
  Hugging Face.

Render и apply:

```bash
kubectl kustomize infra/k8s/overlays/inference
kubectl apply -k infra/k8s/overlays/inference
kubectl -n cadence-md rollout status deployment/vllm-embeddings --timeout=30m
kubectl -n cadence-md rollout status deployment/vllm-reranker --timeout=30m
kubectl -n cadence-md rollout status deployment/vllm-llm --timeout=45m
kubectl -n cadence-md rollout status deployment/inference-gateway
```

### 6. Health checks

Проверьте публичный edge:

```bash
curl -fsS "https://cadence.example.com/nginx-health"
curl -fsS "https://cadence.example.com/api/v1/health/ready"
curl -fsS "https://cadence.example.com/" >/dev/null
```

Проверьте internal services:

```bash
kubectl -n cadence-md get pods
kubectl -n cadence-md logs deployment/backend
kubectl -n cadence-md logs deployment/rag-worker
kubectl -n cadence-md port-forward svc/prometheus 9090:9090
kubectl -n cadence-md port-forward svc/grafana 3000:3000
kubectl -n cadence-md port-forward svc/flower 5555:5555
```

С in-cluster vLLM проверьте gateway из namespace:

```bash
kubectl -n cadence-md run inference-smoke --rm -i --restart=Never \
  --image=python:3.13-slim -- python -c \
  "import urllib.request; print(urllib.request.urlopen('http://inference-gateway:8080/v1/models', timeout=5).read().decode())"
```

### 7. Scaling и updates

Масштабируйте stateless workloads через Deployments:

```bash
kubectl -n cadence-md scale deployment/backend --replicas=3
kubectl -n cadence-md scale deployment/rag-worker --replicas=2
kubectl -n cadence-md scale deployment/frontend --replicas=2
kubectl -n cadence-md scale deployment/nginx --replicas=2
```

Не масштабируйте включённые single-node Postgres, Redis или Qdrant manifests без
замены на HA-дизайн или managed services.

Деплой нового application image:

```bash
cd infra/k8s/overlays/prod
kustomize edit set image cadence-md-backend=<registry>/cadence-md-backend:<tag>
kustomize edit set image cadence-md-rag-worker=<registry>/cadence-md-rag-worker:<tag>
kustomize edit set image cadence-md-frontend=<registry>/cadence-md-frontend:<tag>
cd -

kubectl apply -k infra/k8s/overlays/prod
kubectl -n cadence-md delete job migrations
kubectl apply -k infra/k8s/overlays/prod
kubectl -n cadence-md wait --for=condition=complete job/migrations --timeout=10m
kubectl -n cadence-md rollout status deployment/backend
kubectl -n cadence-md rollout status deployment/rag-worker
kubectl -n cadence-md rollout status deployment/frontend
```

Rollback stateless workloads через Kubernetes rollout history:

```bash
kubectl -n cadence-md rollout undo deployment/backend
kubectl -n cadence-md rollout undo deployment/rag-worker
kubectl -n cadence-md rollout undo deployment/frontend
```

Database migrations не всегда обратимы автоматически. Перед rollback через
изменения схемы проверьте Alembic downgrade steps.

### 8. Backups и security

Делайте backup Postgres PVC или запускайте `pg_dump` из доверенного internal
pod. Для Qdrant делайте backup storage и запускайте snapshots через internal API
с API key из `cadence-md-secret`. `rag-corpus` PVC можно пересоздать через Job
`dvc-pull`, если ваша эксплуатационная политика не требует volume-level backup.

Security checklist:

- Публикуйте наружу только edge nginx Service через Ingress или LoadBalancer.
- Оставляйте Postgres, Redis, Qdrant, backend, frontend, Prometheus, Grafana и
  Flower приватными ClusterIP services, если они не закрыты VPN или admin auth.
- Терминируйте TLS на Ingress или внешнем load balancer.
- Держите `LANGFUSE_TRACE_QUERY_MODE=redacted`, пока отправка полного
  медицинского текста в tracing явно не согласована.
- Храните реальные secrets через Kubernetes Secret manager integration, если она
  доступна; не коммитьте сгенерированный Secret YAML.
