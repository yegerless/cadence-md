# Инструкция по настройке проекта

Следуйте инструкциям ниже, чтобы настроить проект.

### Шаги установки

#### 1. Клонирование репозитория

Склонируйте репозиторий проекта с GitHub на локальную машину:

```bash
git clone git@github.com:CORESIGHT-Health/simple-randomiser.git
cd simple-randomiser
```

#### 2. Инициализация виртуального окружения и установка зависимостей

Выполните следующую команду:

```
poetry install
```

Если нужно установить все зависимости (включая зависимости для разработки),
используйте команду:

```
poetry install --with dev
```

#### 3. Установка pre-commit hook

Инструмент pre-commit уже настроен, вам нужно только установить git-hook:

```bash
poetry run pre-commit install
```

#### 4. Получение данных из удаленного DVC-хранилища

В этом проекте используется объектное хранилище Yandex Cloud как удаленное
DVC-хранилище для хранения текстового корпуса российских клинических
рекомендаций. Базовые настройки подключения к удаленному DVC-хранилищу уже
прописаны в файле `.dvc/config`. Используйте следующие команды, чтобы настроить
параметры безопасности DVC и скачать текстовый корпус на локальный компьютер.

```bash
# Добавьте credentials для подключения к удаленному DVC-хранилищу или используйте .dvc/config.local.example
poetry run dvc remote modify --local yandex access_key_id '<your_access_key_id>'
poetry run dvc remote modify --local yandex secret_access_key '<your_secret_access_key>'

# Скачайте данные из удаленного DVC-хранилища
poetry run dvc pull
```

При запуске через `docker-compose-dev.yml` сервис `dvc-pull` автоматически
скачивает `data.dvc` в общий volume `rag-corpus`. Этот volume монтируется
read-only в `rag-worker` для индексации в Qdrant при необходимости и в `backend`
для авторизованного скачивания PDF-источников.

#### Дополнительная информация

Всегда запускайте pre-commit перед созданием новых коммитов:

```bash
poetry run pre-commit run -av
```

Все проверки pre-commit должны быть зелеными.

Экспорт зависимостей из poetry в `requirements.txt` для запуска проекта в
docker:

```bash
poetry export --without-hashes --format=requirements.txt --without dev > requirements.txt
```

Вы можете использовать следующую команду для генерации QA-валидационного
датасета (нужен GigaChat API Key):

```bash
python commands.py generate-qa

# Используйте для просмотра справки по опциям команды
python commands.py generate-qa --help
```

#### 5. Запуск локальной dev-инфраструктуры

Создайте локальный env-файл из шаблона в репозитории и замените placeholder
значения при необходимости. Не коммитьте `.env.dev`.

```bash
cp .env.example .env.dev
```

Проверьте compose-файл с явным env-файлом:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml config
```

Запустите полный dev-стек. OpenAI-совместимый inference server не входит в этот
compose-файл, поэтому запустите его отдельно заранее и укажите его адрес в
`MODEL_INFERENCE_BASE_URL`.

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up --build
```

Этот compose-поток — основной способ запуска RAG в dev: backend принимает
chat-запросы, а `rag-worker` обрабатывает их асинхронно. Интерактивного
пользовательского REPL entrypoint для RAG больше нет.

Если optional query clarification node требует ответ врача, worker сохраняет
запрос в статусе `awaiting_clarification` без создания финальной строки
`rag_responses`. Frontend poll-ит до этого состояния, показывает форму уточнения
и продолжает тот же запрос через
`POST /api/v1/chat/messages/{request_id}/clarification`. Статус
`awaiting_clarification` не terminal; пользователь всё ещё может отменить
запрос.

Если нужны только инфраструктурные зависимости, запустите их явно:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up postgres redis qdrant
```

Запустите миграции БД вручную при необходимости:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml run --rm migrations
```

В dev compose также есть FastAPI backend и RAG Celery worker. При запуске через
compose `backend` и `rag-worker` ждут успешного выполнения миграций:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up backend
```

Запустите RAG worker, когда нужно обрабатывать асинхронные chat-запросы:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up rag-worker
```

Для локальной наблюдаемости доступны Prometheus и Grafana:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml up prometheus grafana
```

Проверьте health endpoints и метрики:

```bash
curl -fsS http://127.0.0.1:8000/api/v1/health/live
curl -fsS http://127.0.0.1:8000/api/v1/health/ready
curl -fsS http://127.0.0.1:8000/api/v1/health/rag
docker compose --env-file .env.dev -f docker-compose-dev.yml exec rag-worker \
  python -m cadence_md.workers.rag_health --timeout 5
curl -fsS http://127.0.0.1:8000/metrics
curl -fsS http://127.0.0.1:9100/metrics
```

Метрики backend доступны на `http://127.0.0.1:8000/metrics`, метрики worker — на
`http://127.0.0.1:9100/metrics`. Prometheus по умолчанию доступен на
`http://127.0.0.1:9090`, Grafana — на `http://127.0.0.1:3000`. RAG-граф
экспортирует latency узлов `query_rewrite`, `qdrant`, `rerank`, `context`,
`context_relevance`, `llm` и `answer_format`, а также fallback счётчики
`query_rewrite_fallback`, `context_relevance_fallback`,
`answer_format_fallback`, `clarification_required` и уже существующие retrieval
/ generation fallbacks.

React SPA находится в `frontend/` и запускается через frontend profile:

```bash
docker compose --env-file .env.dev -f docker-compose-dev.yml --profile frontend up --build frontend
```

По умолчанию `VITE_API_BASE_URL` пустой, а Vite dev server проксирует
`/api/v1/...` в `VITE_API_PROXY_TARGET` (`http://backend:8000` в compose). Так
локальный браузер работает с same-origin `/api` и не требует CORS middleware в
backend. Для запуска frontend с backend на хосте:

```bash
cd frontend
npm install
VITE_API_BASE_URL= VITE_API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
npm run build
npm test
```

Frontend auth хранит только короткоживущий access token в `sessionStorage`;
refresh token в MVP не используется. Logout и любой API `401` очищают клиентское
auth state и переводят пользователя на `/login`. После изменений backend
проверяйте logout, автоматический redirect на `401`, chat submit/polling,
отправку/отмену уточнения, retry и отображение источников.

#### 5.1 Тестовый стек и команды тестов

Поднимите отдельные зависимости для integration-тестов (тесты сами сервисы не
поднимают):

```bash
docker compose -f docker-compose-test.yml up -d postgres-test redis-test
```

Опциональный Qdrant для smoke-проверок:

```bash
docker compose -f docker-compose-test.yml --profile with-qdrant up -d qdrant-test
```

Smoke-проверка compose-конфигов:

```bash
docker compose -f docker-compose-test.yml config
docker compose -f docker-compose-dev.yml config
```

Быстрые unit-тесты:

```bash
poetry run pytest -m "not integration"
```

Integration и smoke тесты:

```bash
INTEGRATION_DATABASE_URL=postgresql+asyncpg://cadence_md:cadence_md_dev@127.0.0.1:55432/cadence_md_test \
INTEGRATION_REDIS_URL=redis://127.0.0.1:56379/0 \
poetry run pytest -m integration
```

Frontend smoke и build:

```bash
cd frontend
npm test
npm run build
```

Langfuse выключен по умолчанию и считается внешним сервисом для этого dev
compose-файла. Если включаете `LANGFUSE_ENABLED=true`, храните реальные
`LANGFUSE_PUBLIC_KEY` и `LANGFUSE_SECRET_KEY` только в `.env.dev`. Полный
медицинский текст запроса редактируется, пока явно не задан
`LANGFUSE_TRACE_QUERY_MODE=full`.

Metrics-команды (`metrics-eval-full` и `metrics-eval-retriever`) используют тот
же RAG service contract, но отключают user clarification ради batch safety.
Варианты графа проверяются флагами optional nodes `--enable-*` / `--disable-*`;
manifest сохраняет effective profile и additive latency fields.

Inference server остаётся внешним для `docker-compose-dev.yml` и должен быть
доступен по `MODEL_INFERENCE_BASE_URL`.

#### 6. Запуск inference-сервера llama.cpp (embedder, reranker, llm)

Если вы используете Mac, llama.cpp — это единственная возможность запускать
reranker-модели с OpenAI-совместимым API. На Linux используйте vLLM.

Проект использует один OpenAI-совместимый base URL (`MODEL_INFERENCE_BASE_URL`)
для инференса моделей. Для запуска проекта вам нужны:

- embedder;
- reranker;
- LLM для генерации текста.

Названия моделей берутся из `llama-configs/*.ini`. Вы можете использовать
существующий файл или создать свой.

Установите llama.cpp один раз:

```bash
brew install llama.cpp
```

Для скачивания GGUF-файлов с Hugging Face можно использовать `llama-cli`:

```bash
llama-cli --hf-repo repo-owner/hf-repo --hf-file filename.gguf
```

По умолчанию (на Mac) скачанные файлы хранятся в локальном кэше HF:
`~/.cache/huggingface/hub`

Пример команды запуска `llama-server`:

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

Важно:

- `cadence_md` сейчас использует один `MODEL_INFERENCE_BASE_URL` для всех типов
  моделей;
