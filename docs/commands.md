# CADENCE-MD CLI Reference

Project maintenance commands are exposed through [`commands.py`](../commands.py)
and should be run from the repository root with Poetry:

```bash
poetry run python commands.py --help
poetry run python commands.py <subcommand> --help
```

The CLI is for corpus preparation, QA dataset generation, and offline metrics.
User-facing RAG requests are served through the FastAPI chat API and processed
by `rag-worker`; there is no interactive RAG REPL.

For environment setup, Qdrant, DVC, and inference requirements, see
[`docs/setup.md`](setup.md).

## `parse-pdf`

Parses a directory of clinical guideline PDFs into a JSONL file with extracted
`ClinicalSection` records.

| Option          | Default  | Description                                      |
| --------------- | -------- | ------------------------------------------------ |
| `--pdf-dir`     | required | Input directory containing `*.pdf` files.        |
| `--output-file` | required | Output JSONL file with parsed clinical sections. |
| `--max-files`   | none     | Optional cap on the number of input PDFs.        |

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

## `generate-qa`

Generates a synthetic QA dataset from parsed clinical sections.

| Option               | Default                                             | Description                                      |
| -------------------- | --------------------------------------------------- | ------------------------------------------------ |
| `--sections-file`    | `data/clinical_sections.jsonl`                      | Input sections JSONL produced by `parse-pdf`.    |
| `--output-file`      | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | Output QA JSONL file.                            |
| `--model`            | `GigaChat-2-Max`                                    | LLM name used by the QA generator.               |
| `--temperature`      | `0.0`                                               | Sampling temperature for QA generation.          |
| `--max-context`      | `10000`                                             | Maximum section context length in characters.    |
| `--sections-per-pdf` | `3`                                                 | Randomly sample up to N sections per source PDF. |
| `--seed`             | none                                                | Optional random seed for reproducible sampling.  |

Example:

```bash
poetry run python commands.py generate-qa \
  --sections-file data/clinical_sections.jsonl
```

The command requires QA generator credentials such as `GIGACHAT_API_KEY`. It
fails if `--output-file` already exists to avoid appending to stale datasets.

On success, a Markdown generation report is written next to `--output-file`:
`{stem}_generation_report.md`. For example,
`data/metrics_evaluation_datasets/qa_dataset.jsonl` produces
`data/metrics_evaluation_datasets/qa_dataset_generation_report.md`.

The report includes CLI parameters, loaded sections, skipped invalid JSONL rows,
generated pairs, failed sections, and distributions by question type and section
type.

## `metrics-eval-full`

Runs full RAG evaluation: answer generation, RAGAS metrics, retrieval metrics,
and Markdown/JSON/Parquet artifacts.

| Option                                                                     | Default                                             | Description                                                       |
| -------------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------- |
| `--dataset-file`                                                           | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | Evaluation QA JSONL file.                                         |
| `--output-dir`                                                             | `metrics/results/`                                  | Directory for run artifacts.                                      |
| `--sample-size`                                                            | none                                                | Optional cap on the number of test cases.                         |
| `--seed`                                                                   | none                                                | Optional random seed for reproducible `--sample-size` sampling.   |
| `--k`                                                                      | `5`                                                 | K for recall@K and precision@K.                                   |
| `--enable-text-matcher-metrics`                                            | off                                                 | Also compute diagnostic `text_match_*` metrics.                   |
| `--enable-input-guardrails` / `--disable-input-guardrails`                 | settings default                                    | Override optional input guardrails for this metrics run.          |
| `--enable-query-rewriter` / `--disable-query-rewriter`                     | settings default                                    | Override optional query rewriting for this metrics run.           |
| `--enable-context-relevance-grader` / `--disable-context-relevance-grader` | settings default                                    | Override optional context relevance grading for this metrics run. |
| `--enable-answer-formatter` / `--disable-answer-formatter`                 | settings default                                    | Override optional answer formatting for this metrics run.         |
| `--enable-output-guardrails` / `--disable-output-guardrails`               | settings default                                    | Override optional output guardrails for this metrics run.         |
| `--enable-query-clarification` / `--disable-query-clarification`           | settings default                                    | Override optional query clarification for this metrics run.       |
| `--max-query-rewrite-iterations`                                           | settings default                                    | Override the maximum context relevance rewrite iterations.        |

Example:

```bash
poetry run python commands.py metrics-eval-full --sample-size 20 --seed 42
```

The command expects a running Qdrant instance, an indexed corpus, and an
available OpenAI-compatible inference server. Offline metrics runs pass
`allow_clarification=false` to the RAG service so batch evaluation cannot pause
waiting for human input. Optional input guardrails run before retrieval and can
short-circuit non-medical cases with a template response; use
`--disable-input-guardrails` when evaluating datasets that should bypass this
classifier. RAGAS GigaChat embeddings are capped by
`GIGACHAT_EMBEDDINGS_MAX_TEXT_CHARS` and `GIGACHAT_EMBEDDINGS_MAX_BATCH_CHARS`
to avoid oversized payload errors on long contexts or answers.

## `metrics-eval-retriever`

Runs retriever-only evaluation: retrieve + rerank, without RAGAS or answer
generation.

| Option                                                                     | Default                                             | Description                                                       |
| -------------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------- |
| `--dataset-file`                                                           | `data/metrics_evaluation_datasets/qa_dataset.jsonl` | Evaluation QA JSONL file.                                         |
| `--output-dir`                                                             | `metrics/results/`                                  | Directory for run artifacts.                                      |
| `--sample-size`                                                            | none                                                | Optional cap on the number of test cases.                         |
| `--seed`                                                                   | none                                                | Optional random seed for reproducible `--sample-size` sampling.   |
| `--k`                                                                      | pipeline default                                    | K for recall@K and precision@K.                                   |
| `--workers`                                                                | `1`                                                 | Parallel worker threads for independent retrieve + rerank cases.  |
| `--enable-text-matcher-metrics`                                            | off                                                 | Also compute diagnostic `text_match_*` metrics.                   |
| `--enable-input-guardrails` / `--disable-input-guardrails`                 | settings default                                    | Override optional input guardrails for this metrics run.          |
| `--enable-query-rewriter` / `--disable-query-rewriter`                     | settings default                                    | Override optional query rewriting for this metrics run.           |
| `--enable-context-relevance-grader` / `--disable-context-relevance-grader` | settings default                                    | Override optional context relevance grading for this metrics run. |
| `--enable-answer-formatter` / `--disable-answer-formatter`                 | settings default                                    | Override optional answer formatting for this metrics run.         |
| `--enable-output-guardrails` / `--disable-output-guardrails`               | settings default                                    | Override optional output guardrails for this metrics run.         |
| `--enable-query-clarification` / `--disable-query-clarification`           | settings default                                    | Override optional query clarification for this metrics run.       |
| `--max-query-rewrite-iterations`                                           | settings default                                    | Override the maximum context relevance rewrite iterations.        |

Example:

```bash
poetry run python commands.py metrics-eval-retriever \
  --sample-size 100 \
  --seed 42 \
  --k 5 \
  --workers 4
```

## Metrics Artifacts

Both metrics commands create a dedicated run directory under `--output-dir`:

```text
metrics/results/<run_id>/
  run_manifest.json
  summary_metrics.json
  report.md
  errors.jsonl
  # full mode
  cases.jsonl
  ragas_scores.parquet
  # retriever mode
  retrieval_cases.jsonl
```

Primary retrieval metrics are section-based and require `section_id` in both the
QA dataset and indexed Qdrant metadata. Regenerate the QA dataset and reindex
the corpus after section metadata changes. Use `--enable-text-matcher-metrics`
only for diagnostic text-overlap metrics; they are written separately as
`text_match_*`.

Run artifacts include the effective optional-node profile and additive latency
fields when present: `input_guardrails`, `query_rewrite`, `qdrant`, `rerank`,
`context_relevance`, `llm`, `answer_format`, `output_guardrails`, and
`total_ms`. The graph profile reports `input_guardrails` as effective in both
full and retriever modes when enabled, because it runs before retrieval. When
`--seed` is provided, `run_manifest.json` and `report.md` record it as
`sample_seed` so sampled runs can be repeated.
