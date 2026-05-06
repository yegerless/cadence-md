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
`http://127.0.0.1:9090`, Grafana — на `http://127.0.0.1:3000`.

Frontend пока не включён, потому что приложение `frontend/` ещё не добавлено в
репозиторий. В `docker-compose-dev.yml` оставлен закомментированный placeholder
для будущего frontend profile.

Langfuse выключен по умолчанию и считается внешним сервисом для этого dev
compose-файла. Если включаете `LANGFUSE_ENABLED=true`, храните реальные
`LANGFUSE_PUBLIC_KEY` и `LANGFUSE_SECRET_KEY` только в `.env.dev`. Полный
медицинский текст запроса редактируется, пока явно не задан
`LANGFUSE_TRACE_QUERY_MODE=full`.

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
