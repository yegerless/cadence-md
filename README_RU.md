# CADENCE-MD 🏥

**C**linical **A**ssistant for **D**iagnosis, **E**vidence **N**avigation &
**C**ase **E**valuation for **M**edical **D**octors

> Интеллектуальная система поддержки врачебных решений

## Обзор

CADENCE-MD — это AI-ассистент для врачей, помогающий принимать доказательные
решения путем предоставления мгновенного доступа к клиническим рекомендациям и
новейшим медицинским исследованиям через продвинутую RAG-архитектуру и
интеграцию с PubMed.

## Ключевые возможности

- **RAG по клиническим рекомендациям**: Семантический поиск по медицинским
  протоколам и руководствам
- **Интеграция с PubMed**: Доступ к актуальным медицинским исследованиям
- **Поддержка диагностики**: AI-помощник в дифференциальной диагностике и
  подборе терапии
- **Ориентация на врачей**: Разработано для интеграции в клиническую практику

## Технологический стек

- LLM + RAG архитектура
- Векторные базы данных и семантический поиск
- Интеграция с PubMed API
- Медицинская NLP обработка

## Команды CLI

Сценарии запуска собраны в [`commands.py`](commands.py). Из корня репозитория:

```bash
poetry run python commands.py --help
poetry run python commands.py <подкоманда> --help
```

### `parse-pdf`

Парсит директорию с клиническими рекомендациями в PDF и сохраняет результат в
JSONL с секциями (`ClinicalSection`).

| Опция           | По умолчанию | Описание                                     |
| --------------- | ------------ | -------------------------------------------- |
| `--pdf-dir`     | —            | Входная директория с файлами `*.pdf`         |
| `--output-file` | —            | Выходной JSONL-файл с распарсенными секциями |
| `--max-files`   | —            | Опционально: ограничить число входных PDF    |

Примеры:

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

Генерация синтетического QA-датасета из JSONL с клиническими секциями (результат
парсера).

| Опция                | По умолчанию                                        | Описание                                                        |
| -------------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| `--sections-file`    | `data/clinical_sections.jsonl`                      | Входной JSONL секций                                            |
| `--output-file`      | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | Выходной QA JSONL                                               |
| `--model`            | `GigaChat-2-Max`                                    | Имя LLM (например `GigaChat`, `GigaChat-2-Max`, `GigaChat-pro`) |
| `--temperature`      | `0.0`                                               | Температура сэмплирования                                       |
| `--max-context`      | `10000`                                             | Максимальная длина контекста (символы)                          |
| `--sections-per-pdf` | `3`                                                 | Случайно выбрать до N секций на каждый исходный PDF             |
| `--seed`             | —                                                   | Seed для воспроизводимой случайной выборки                      |

Пример:

```bash
poetry run python commands.py generate-qa --sections-file data/clinical_sections.jsonl
```

После успешной генерации рядом с `--output-file` создаётся Markdown-отчёт:
`{stem}_generation_report.md`, где `{stem}` — имя выходного файла без расширения
(например `.../qa_dataset.jsonl` → `.../qa_dataset_generation_report.md`). В
отчёте: параметры запуска, сводка конвейера (загрузка секций, пропуски битых
строк JSONL, число сгенерированных пар и неудачных секций), распределения по
типу вопроса и типу секции.

Нужны учётные данные LLM (например `GIGACHAT_API_KEY`), как настроено у
генератора QA. Команда завершается с ошибкой, если файл `--output-file` уже
существует.

### `metrics-eval-full`

Полная оценка RAG: генерация ответов, метрики RAGAS, метрики ретрива и отчёты в
каталоге вывода.

| Опция                           | По умолчанию                                        | Описание                                                     |
| ------------------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| `--dataset-file`                | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | JSONL с вопросами для оценки                                 |
| `--output-dir`                  | `metrics/results/`                                  | Отчёты и файлы метрик                                        |
| `--sample-size`                 | —                                                   | Ограничить число тест-кейсов                                 |
| `--k`                           | `5`                                                 | K для recall@K / precision@K                                 |
| `--enable-text-matcher-metrics` | выкл.                                               | Дополнительно считать диагностические `text_match_*` метрики |

Пример:

```bash
poetry run python commands.py metrics-eval-full --sample-size 20
```

Нужны запущенный Qdrant, проиндексированный корпус и остальное окружение по
инструкции развёртывания.

### `metrics-eval-retriever`

Оценка только ретривера (поиск + реранк): без RAGAS и без полной генерации
ответа.

Те же опции, что у `metrics-eval-full`, плюс:

| Опция       | По умолчанию       | Описание                                                          |
| ----------- | ------------------ | ----------------------------------------------------------------- |
| `--k`       | значение пайплайна | K для recall@K / precision@K                                      |
| `--workers` | `1`                | Параллельные воркер-потоки для независимых кейсов retrieve+rerank |

Оба режима (`full` и `retriever`) создают отдельную директорию прогона в
`--output-dir`:

```text
metrics/results/<run_id>/
  run_manifest.json
  summary_metrics.json
  report.md
  errors.jsonl
  # режим full
  cases.jsonl
  ragas_scores.parquet
  # режим retriever
  retrieval_cases.jsonl
```

Основные метрики ретривера теперь считаются по `section_id`, поэтому для точной
оценки нужны новый QA-датасет и переиндексация корпуса в Qdrant. Флаг
`--enable-text-matcher-metrics` включает только дополнительные диагностические
метрики по текстовому overlap и сохраняет их отдельно как `text_match_*`.

Пример:

```bash
poetry run python commands.py metrics-eval-retriever --k 5
```

Список флагов: `poetry run python commands.py metrics-eval-full --help` и
`metrics-eval-retriever --help`.

## Health сервиса

FastAPI backend отдаёт versioned health-пробы для локального Docker Compose и
дальнейшей оркестрации:

- `GET /api/v1/health/live` проверяет только то, что процесс backend жив.
- `GET /api/v1/health/ready` проверяет Postgres, Redis, Qdrant
  (доступность+схему коллекции) и OpenAI-compatible inference API. При
  недоступности обязательной зависимости возвращает `503`.
- `GET /api/v1/health/rag` возвращает детальный статус RAG-зависимостей (Qdrant,
  inference, queue broker) без раскрытия API keys и других секретов.

Политика graceful degradation: при старте backend не поднимает тяжёлый RAG stack
и не падает, если Qdrant или inference временно недоступны. В таком состоянии
readiness возвращает `503`, чтобы Docker/оркестратор не направлял трафик. При
этом chat API остаётся асинхронным: `POST /api/v1/chat/messages` может сохранить
и поставить запрос в очередь, если доступны Postgres, Redis и Celery; если на
этапе выполнения worker не сможет достучаться до Qdrant или inference, запрос
завершится контролируемой ошибкой.

## Лицензия

Проприетарная - подробности в [LICENSE_RU.md](LICENSE_RU.md).

---

**Важно**: Это исследовательский проект. Не предназначен для прямого
клинического использования без валидации.
