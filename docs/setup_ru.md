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

#### 5. Запуск inference-сервера llama.cpp (embedder, reranker, llm)

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
