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
| `--base-url`         | —                                                   | Optional base URL for a local model                          |
| `--load-api-key`     | on                                                  | Load API key from the environment                            |
| `--no-load-api-key`  | off                                                 | Disable loading API key from the environment                 |
| `--temperature`      | `0.0`                                               | Sampling temperature                                         |
| `--max-context`      | `10000`                                             | Max context length (characters)                              |
| `--sections-per-pdf` | `3`                                                 | Randomly sample up to N sections from each source PDF        |
| `--seed`             | —                                                   | Random seed for reproducible section sampling                |

Example:

```bash
poetry run python commands.py generate-qa --sections-file data/clinical_sections.jsonl
```

Requires LLM credentials (e.g. `GIGACHAT_API_KEY`) as configured for the QA
generator. The command fails if `--output-file` already exists to avoid
accidental appends to stale datasets.

### `metrics-eval-full`

Full RAG evaluation: answer generation, RAGAS metrics, retrieval metrics, and
reports under the output directory.

| Option           | Default                                             | Description                    |
| ---------------- | --------------------------------------------------- | ------------------------------ |
| `--dataset-file` | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | Evaluation QA JSONL            |
| `--output-dir`   | `metrics/results/`                                  | Reports and metric files       |
| `--sample-size`  | —                                                   | Limit the number of test cases |
| `--pdf-dir`      | `data/main_specialities/`                           | PDF directory for Qdrant setup |

Example:

```bash
poetry run python commands.py metrics-eval-full --sample-size 20
```

Expects a running Qdrant instance, an indexed corpus, and the rest of the stack
described in the setup guide.

### `metrics-eval-retriever`

Retriever-only evaluation (retrieve + rerank): no RAGAS and no full answer
generation. Writes `retriever_evaluation_report.md` and `retriever_metrics.json`
into the output directory.

Same options as `metrics-eval-full`, plus:

| Option | Default          | Description                  |
| ------ | ---------------- | ---------------------------- |
| `--k`  | pipeline default | K for recall@K / precision@K |

Example:

```bash
poetry run python commands.py metrics-eval-retriever --k 5
```

### Direct `metrics/main.py` entrypoint

You can also run the metrics CLI directly (same subcommands and flags):

```bash
poetry run python metrics/main.py full --help
poetry run python metrics/main.py retriever --help
```

## License

Proprietary - See [LICENSE.md](LICENSE.md) for details.

---

**Disclaimer**: This is a research project. Not for direct clinical use without
validation.
