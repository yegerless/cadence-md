# CLI-справочник CADENCE-MD

Проектные команды доступны через [`commands.py`](../commands.py). Запускайте их
из корня репозитория через Poetry:

```bash
poetry run python commands.py --help
poetry run python commands.py <subcommand> --help
```

CLI используется для подготовки корпуса, генерации QA-датасета и offline
metrics. Пользовательские RAG-запросы обслуживаются через FastAPI chat API и
обрабатываются `rag-worker`; интерактивного RAG REPL нет.

Требования к окружению, Qdrant, DVC и inference описаны в
[`docs/setup_ru.md`](setup_ru.md).

## `parse-pdf`

Парсит директорию с PDF клинических рекомендаций в JSONL-файл с записями
`ClinicalSection`.

| Опция           | По умолчанию | Описание                                         |
| --------------- | ------------ | ------------------------------------------------ |
| `--pdf-dir`     | обязательно  | Входная директория с файлами `*.pdf`.            |
| `--output-file` | обязательно  | Выходной JSONL-файл с клиническими секциями.     |
| `--max-files`   | нет          | Опциональное ограничение количества входных PDF. |

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

## `generate-qa`

Генерирует синтетический QA-датасет из распарсенных клинических секций.

| Опция                | По умолчанию                                        | Описание                                        |
| -------------------- | --------------------------------------------------- | ----------------------------------------------- |
| `--sections-file`    | `data/clinical_sections.jsonl`                      | Входной JSONL секций, созданный `parse-pdf`.    |
| `--output-file`      | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | Выходной QA JSONL-файл.                         |
| `--model`            | `GigaChat-2-Max`                                    | Имя LLM для генератора QA.                      |
| `--temperature`      | `0.0`                                               | Температура сэмплирования при генерации QA.     |
| `--max-context`      | `10000`                                             | Максимальная длина контекста секции в символах. |
| `--sections-per-pdf` | `3`                                                 | Случайно выбрать до N секций на исходный PDF.   |
| `--seed`             | нет                                                 | Опциональный seed для воспроизводимой выборки.  |

Пример:

```bash
poetry run python commands.py generate-qa \
  --sections-file data/clinical_sections.jsonl
```

Команде нужны credentials генератора QA, например `GIGACHAT_API_KEY`. Если
`--output-file` уже существует, команда завершается ошибкой, чтобы не дописать
данные в устаревший датасет.

После успешной генерации рядом с `--output-file` создаётся Markdown-отчёт:
`{stem}_generation_report.md`. Например,
`data/metrics_evaluation_datasets/qa_dataset.jsonl` создаёт
`data/metrics_evaluation_datasets/qa_dataset_generation_report.md`.

В отчёте сохраняются CLI-параметры, загруженные секции, пропущенные битые строки
JSONL, сгенерированные пары, неудачные секции и распределения по типам вопросов
и секций.

## `metrics-eval-full`

Запускает полную оценку RAG: генерацию ответов, RAGAS metrics, retrieval metrics
и Markdown/JSON/Parquet-артефакты.

| Опция                                                                      | По умолчанию                                        | Описание                                                                 |
| -------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------ |
| `--dataset-file`                                                           | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | QA JSONL-файл для оценки.                                                |
| `--output-dir`                                                             | `metrics/results/`                                  | Директория для артефактов прогона.                                       |
| `--sample-size`                                                            | нет                                                 | Опциональное ограничение числа тест-кейсов.                              |
| `--seed`                                                                   | нет                                                 | Опциональный seed для воспроизводимой выборки `--sample-size`.           |
| `--k`                                                                      | `5`                                                 | K для recall@K и precision@K.                                            |
| `--enable-text-matcher-metrics`                                            | выкл.                                               | Дополнительно считать диагностические `text_match_*` метрики.            |
| `--enable-input-guardrails` / `--disable-input-guardrails`                 | settings default                                    | Переопределить optional input guardrails для этого metrics run.          |
| `--enable-query-rewriter` / `--disable-query-rewriter`                     | settings default                                    | Переопределить optional query rewriting для этого metrics run.           |
| `--enable-context-relevance-grader` / `--disable-context-relevance-grader` | settings default                                    | Переопределить optional context relevance grading для этого metrics run. |
| `--enable-answer-formatter` / `--disable-answer-formatter`                 | settings default                                    | Переопределить optional answer formatting для этого metrics run.         |
| `--enable-output-guardrails` / `--disable-output-guardrails`               | settings default                                    | Переопределить optional output guardrails для этого metrics run.         |
| `--enable-query-clarification` / `--disable-query-clarification`           | settings default                                    | Переопределить optional query clarification для этого metrics run.       |
| `--max-query-rewrite-iterations`                                           | settings default                                    | Переопределить максимум context relevance rewrite iterations.            |

Пример:

```bash
poetry run python commands.py metrics-eval-full --sample-size 20 --seed 42
```

Команде нужны запущенный Qdrant, проиндексированный корпус и доступный
OpenAI-compatible inference server. Offline metrics runs передают в RAG service
`allow_clarification=false`, чтобы batch-оценка не останавливалась в ожидании
ответа человека. Optional input guardrails запускаются до retrieval и могут
завершить немедицинский кейс шаблонным ответом; используйте
`--disable-input-guardrails`, если оценочный датасет должен обходить этот
классификатор. GigaChat embeddings для RAGAS ограничиваются
`GIGACHAT_EMBEDDINGS_MAX_TEXT_CHARS` и `GIGACHAT_EMBEDDINGS_MAX_BATCH_CHARS`,
чтобы длинные контексты или ответы не падали с oversized payload.

## `metrics-eval-retriever`

Запускает retriever-only оценку: retrieve + rerank, без RAGAS и без генерации
ответа.

| Опция                                                                      | По умолчанию                                        | Описание                                                                 |
| -------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------ |
| `--dataset-file`                                                           | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | QA JSONL-файл для оценки.                                                |
| `--output-dir`                                                             | `metrics/results/`                                  | Директория для артефактов прогона.                                       |
| `--sample-size`                                                            | нет                                                 | Опциональное ограничение числа тест-кейсов.                              |
| `--seed`                                                                   | нет                                                 | Опциональный seed для воспроизводимой выборки `--sample-size`.           |
| `--k`                                                                      | pipeline default                                    | K для recall@K и precision@K.                                            |
| `--workers`                                                                | `1`                                                 | Параллельные потоки для независимых retrieve + rerank кейсов.            |
| `--enable-text-matcher-metrics`                                            | выкл.                                               | Дополнительно считать диагностические `text_match_*` метрики.            |
| `--enable-input-guardrails` / `--disable-input-guardrails`                 | settings default                                    | Переопределить optional input guardrails для этого metrics run.          |
| `--enable-query-rewriter` / `--disable-query-rewriter`                     | settings default                                    | Переопределить optional query rewriting для этого metrics run.           |
| `--enable-context-relevance-grader` / `--disable-context-relevance-grader` | settings default                                    | Переопределить optional context relevance grading для этого metrics run. |
| `--enable-answer-formatter` / `--disable-answer-formatter`                 | settings default                                    | Переопределить optional answer formatting для этого metrics run.         |
| `--enable-output-guardrails` / `--disable-output-guardrails`               | settings default                                    | Переопределить optional output guardrails для этого metrics run.         |
| `--enable-query-clarification` / `--disable-query-clarification`           | settings default                                    | Переопределить optional query clarification для этого metrics run.       |
| `--max-query-rewrite-iterations`                                           | settings default                                    | Переопределить максимум context relevance rewrite iterations.            |

Пример:

```bash
poetry run python commands.py metrics-eval-retriever \
  --sample-size 100 \
  --seed 42 \
  --k 5 \
  --workers 4
```

## Артефакты metrics

Обе metrics-команды создают отдельную директорию прогона внутри `--output-dir`:

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

Основные retrieval metrics считаются по `section_id`, поэтому `section_id`
должен быть и в QA-датасете, и в Qdrant metadata. После изменений section
metadata нужно заново сгенерировать QA-датасет и переиндексировать корпус. Флаг
`--enable-text-matcher-metrics` нужен только для диагностических text-overlap
метрик; они записываются отдельно как `text_match_*`.

Артефакты прогона содержат effective optional-node profile и additive latency
fields, если они есть: `input_guardrails`, `query_rewrite`, `qdrant`, `rerank`,
`context_relevance`, `llm`, `answer_format`, `output_guardrails` и `total_ms`.
Graph profile показывает `input_guardrails` как effective и в full, и в
retriever mode, если узел включен, потому что он запускается до retrieval. Если
передан `--seed`, `run_manifest.json` и `report.md` сохраняют его как
`sample_seed`, чтобы сэмплированный прогон можно было повторить.
