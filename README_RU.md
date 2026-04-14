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

### `generate-qa`

Генерация синтетического QA-датасета из JSONL с клиническими секциями (результат
парсера).

| Опция                | По умолчанию                                        | Описание                                                        |
| -------------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| `--sections-file`    | `data/clinical_sections.jsonl`                      | Входной JSONL секций                                            |
| `--output-file`      | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | Выходной QA JSONL                                               |
| `--model`            | `GigaChat-2-Max`                                    | Имя LLM (например `GigaChat`, `GigaChat-2-Max`, `GigaChat-pro`) |
| `--base-url`         | —                                                   | Опционально: base URL для локальной модели                      |
| `--load-api-key`     | вкл.                                                | Брать API-ключ из окружения                                     |
| `--no-load-api-key`  | выкл.                                               | Отключить загрузку API-ключа из окружения                       |
| `--temperature`      | `0.0`                                               | Температура сэмплирования                                       |
| `--max-context`      | `10000`                                             | Максимальная длина контекста (символы)                          |
| `--sections-per-pdf` | `3`                                                 | Случайно выбрать до N секций на каждый исходный PDF             |
| `--seed`             | —                                                   | Seed для воспроизводимой случайной выборки                      |

Пример:

```bash
poetry run python commands.py generate-qa --sections-file data/clinical_sections.jsonl
```

Нужны учётные данные LLM (например `GIGACHAT_API_KEY`), как настроено у
генератора QA. Команда завершается с ошибкой, если файл `--output-file` уже
существует.

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

### `metrics-eval-full`

Полная оценка RAG: генерация ответов, метрики RAGAS, метрики ретрива и отчёты в
каталоге вывода.

| Опция            | По умолчанию                                        | Описание                     |
| ---------------- | --------------------------------------------------- | ---------------------------- |
| `--dataset-file` | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | JSONL с вопросами для оценки |
| `--output-dir`   | `metrics/results/`                                  | Отчёты и файлы метрик        |
| `--sample-size`  | —                                                   | Ограничить число тест-кейсов |
| `--pdf-dir`      | `data/main_specialities/`                           | Папка PDF для setup Qdrant   |

Пример:

```bash
poetry run python commands.py metrics-eval-full --sample-size 20
```

Нужны запущенный Qdrant, проиндексированный корпус и остальное окружение по
инструкции развёртывания.

### `metrics-eval-retriever`

Оценка только ретривера (поиск + реранк): без RAGAS и без полной генерации
ответа. В каталоге вывода создаются `retriever_evaluation_report.md` и
`retriever_metrics.json`.

Те же опции, что у `metrics-eval-full`, плюс:

| Опция | По умолчанию       | Описание                     |
| ----- | ------------------ | ---------------------------- |
| `--k` | значение пайплайна | K для recall@K / precision@K |

Пример:

```bash
poetry run python commands.py metrics-eval-retriever --k 5
```

### Прямой запуск `metrics/main.py`

Тот же CLI метрик можно вызвать напрямую (подкоманды `full` и `retriever`, те же
флаги):

```bash
poetry run python metrics/main.py full --help
poetry run python metrics/main.py retriever --help
```

## Лицензия

Проприетарная - подробности в [LICENSE_RU.md](LICENSE_RU.md).

---

**Важно**: Это исследовательский проект. Не предназначен для прямого
клинического использования без валидации.
