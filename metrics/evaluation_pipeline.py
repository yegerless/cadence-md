import json
import logging
import random
import re
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from ragas import evaluate
from ragas.embeddings import (
    LangchainEmbeddingsWrapper,
)
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
)
from ragas.run_config import RunConfig
from tqdm import tqdm

from cadence_md.app.rag import RAGPipeline, RAGState
from cadence_md.app.settings import settings
from metrics.artifacts import (
    append_jsonl,
    build_run_directory,
    create_run_manifest,
    document_to_record,
    write_json,
)
from metrics.report_renderer import render_validation_report
from metrics.retrieval_utils import collect_missed_retrieval_case_ids
from metrics.schemas import QATestCase, RAGTestResult
from metrics.summary_builder import build_summary_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluationPipeline:
    """ """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        gigachat_llm: GigaChat,
        gigachat_embeddings: GigaChatEmbeddings,
        ragas_metrics: list | None = None,
    ):
        """ """
        self.rag_pipeline = rag_pipeline

        # Initialize evaluation LLM and embeddings for ragas metrics
        self.evaluation_llm = LangchainLLMWrapper(gigachat_llm)
        self.evaluation_embeddings = LangchainEmbeddingsWrapper(gigachat_embeddings)

        if ragas_metrics is None:
            self.ragas_metrics = [
                AnswerCorrectness(
                    llm=self.evaluation_llm,
                    embeddings=self.evaluation_embeddings,
                ),
                AnswerRelevancy(
                    llm=self.evaluation_llm,
                    embeddings=self.evaluation_embeddings,
                ),
                Faithfulness(
                    llm=self.evaluation_llm,
                ),
                ContextPrecision(
                    llm=self.evaluation_llm,
                ),
                ContextRecall(
                    llm=self.evaluation_llm,
                ),
                ContextRelevance(
                    llm=self.evaluation_llm,
                ),
            ]
        else:
            self.ragas_metrics = ragas_metrics

        # Run configuration for prevent rate limit errors with GigaChat API
        self.run_config = RunConfig(
            max_workers=1,
            timeout=300,
            max_retries=15,
            max_wait=120,
            log_tenacity=True,
        )

        logger.info(f"Initialized evaluation pipeline with {len(self.ragas_metrics)} metrics")

    def load_test_cases(self, dataset_file: Path) -> list[QATestCase]:
        """Loading test cases from JSONL"""
        test_cases: list[QATestCase] = []

        with Path(dataset_file).open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        test_cases.append(QATestCase.from_dict(data))
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        logger.warning(
                            "Skipping malformed QA row at %s:%s (%s)",
                            dataset_file,
                            line_num,
                            exc,
                        )

        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases

    def run_rag_pipeline(
        self,
        test_cases: list[QATestCase],
        sample_size: int | None = None,
    ) -> list[RAGTestResult]:
        """Running the RAG pipeline for all test cases"""

        if sample_size and sample_size < len(test_cases):
            test_cases = random.sample(test_cases, sample_size)
            logger.info(f"A sample of {sample_size} cases is used")

        results: list[RAGTestResult] = []

        logger.info(f"Running the RAG pipeline for {len(test_cases)} cases...")

        for idx, test_case in enumerate(tqdm(test_cases, desc="RAG inference")):
            try:
                rag_result = self.rag_pipeline.run(test_case.question)

                retrieved_contexts = rag_result["retrieved_docs"]
                retrieved_scores = rag_result["retrieved_scores"]
                generated_answer = rag_result["answer"]

                result = RAGTestResult(
                    question=test_case.question,
                    ground_truth_answer=test_case.answer,
                    ground_truth_context=test_case.context,
                    retrieved_contexts=retrieved_contexts,
                    retrieval_scores=retrieved_scores,
                    generated_answer=generated_answer,
                    question_type=test_case.question_type,
                    section_type=test_case.section_type,
                    test_case_id=idx,
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Case processing error {idx}: {e}")
                continue

        logger.info(f"✓ Processed {len(results)} cases")
        return results

    def run_retriever_pipeline(
        self,
        test_cases: list[QATestCase],
        sample_size: int | None = None,
    ) -> list[RAGTestResult]:
        """Retrieve + rerank only; no LLM generation (retriever-focused evaluation)."""
        if sample_size and sample_size < len(test_cases):
            test_cases = random.sample(test_cases, sample_size)
            logger.info(f"A sample of {sample_size} cases is used")

        results: list[RAGTestResult] = []
        logger.info(f"Running retriever (retrieve + rerank) for {len(test_cases)} cases...")

        for idx, test_case in enumerate(tqdm(test_cases, desc="Retriever")):
            try:
                initial_state: RAGState = {
                    "query": test_case.question,
                    "retrieved_docs": [],
                    "retrieved_scores": [],
                    "reranked_scores": [],
                    "context": "",
                    "context_chars": 0,
                    "answer": "",
                    "answer_word_count": 0,
                }
                state = self.rag_pipeline.retrieve_node(initial_state)
                state = self.rag_pipeline.reranker_node(state)

                retrieved_contexts = state["retrieved_docs"]
                retrieval_scores = (
                    state["reranked_scores"]
                    if state.get("reranked_scores")
                    else state["retrieved_scores"]
                )

                result = RAGTestResult(
                    question=test_case.question,
                    ground_truth_answer=test_case.answer,
                    ground_truth_context=test_case.context,
                    retrieved_contexts=retrieved_contexts,
                    retrieval_scores=retrieval_scores,
                    generated_answer="",
                    question_type=test_case.question_type,
                    section_type=test_case.section_type,
                    test_case_id=idx,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Case processing error {idx}: {e}")
                continue

        logger.info(f"✓ Processed {len(results)} cases (retriever only)")
        return results

    def convert_to_ragas_format(self, results: list[RAGTestResult]) -> Dataset:
        """Converting results to RAGAS format"""
        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for result in results:
            ragas_data["question"].append(result.question)
            ragas_data["answer"].append(result.generated_answer)

            ctxs = result.retrieved_contexts
            ragas_data["contexts"].append([c.page_content for c in ctxs])

            ragas_data["ground_truth"].append(result.ground_truth_answer)

        return Dataset.from_dict(ragas_data)

    def evaluate_with_ragas(self, results: list[RAGTestResult]) -> pd.DataFrame:
        """Assessing results using RAGAS metrics (legacy evaluate API)"""
        if not results:
            logger.warning("No RAG results provided for RAGAS evaluation")
            return pd.DataFrame(columns=["question_type", "section_type", "test_case_id"])

        logger.info("Converting to RAGAS format...")
        ragas_dataset = self.convert_to_ragas_format(results)

        logger.info(f"Running RAGAS scores ({len(self.ragas_metrics)} metrics)...")
        # llm/embeddings are already integrated into the metrics
        ragas_results = evaluate(
            ragas_dataset,
            metrics=self.ragas_metrics,
            return_executor=False,
            run_config=self.run_config,
        )

        df: pd.DataFrame = ragas_results.to_pandas()

        df["question_type"] = [r.question_type for r in results]
        df["section_type"] = [r.section_type for r in results]
        df["test_case_id"] = [r.test_case_id for r in results]

        return df

    @staticmethod
    def _extract_numeric_ragas_scores(ragas_row: pd.Series) -> dict[str, float]:
        """Extract only numeric RAGAS scores from a mixed row."""
        metadata_columns = {
            "question",
            "answer",
            "contexts",
            "ground_truth",
            "question_type",
            "section_type",
            "test_case_id",
        }
        numeric_scores: dict[str, float] = {}
        for column, value in ragas_row.items():
            if column in metadata_columns:
                continue
            try:
                numeric_value = pd.to_numeric(value, errors="coerce")
            except (TypeError, ValueError):
                continue
            if not pd.api.types.is_scalar(numeric_value):
                continue
            if pd.isna(numeric_value):
                continue
            numeric_scores[column] = float(numeric_value)
        return numeric_scores

    def calculate_retrieval_metrics(
        self,
        results: list[RAGTestResult],
        k: int | None = None,
    ) -> dict[str, float | int]:
        """
        Calculation of retriever metrics.
        """
        k = k if k is not None else settings.rag_config.retrieval.dense_top_k
        if k <= 0:
            raise ValueError("k must be a positive integer")

        metrics: dict[str, float | int] = {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "avg_score": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "k": k,
        }

        if not results:
            return metrics

        hits = 0
        reciprocal_ranks: list[float] = []
        top_1_scores: list[float] = []
        recall_at_k_values: list[float] = []
        precision_at_k_values: list[float] = []

        for result in results:
            gt_context_normalized = result.ground_truth_context.lower().strip()
            found = False

            for i, retrieved_ctx in enumerate(result.retrieved_contexts):
                retrieved_normalized = retrieved_ctx.page_content.lower().strip()

                if self.contexts_match(gt_context_normalized, retrieved_normalized):
                    hits += 1
                    reciprocal_ranks.append(1.0 / (i + 1))
                    found = True
                    break

            if not found:
                reciprocal_ranks.append(0.0)

            if result.retrieval_scores:
                top_1_scores.append(result.retrieval_scores[0])

            top_k_docs = result.retrieved_contexts[:k]
            relevant_in_top_k = sum(
                1
                for doc in top_k_docs
                if self.contexts_match(gt_context_normalized, doc.page_content.lower().strip())
            )
            recall_at_k_values.append(1.0 if relevant_in_top_k > 0 else 0.0)
            precision_at_k_values.append((relevant_in_top_k / k) if k > 0 else 0.0)

        metrics["hit_rate"] = hits / len(results)
        metrics["mrr"] = statistics.mean(reciprocal_ranks)
        metrics["avg_score"] = statistics.mean(top_1_scores) if top_1_scores else 0.0
        metrics["recall_at_k"] = statistics.mean(recall_at_k_values)
        metrics["precision_at_k"] = statistics.mean(precision_at_k_values)

        return metrics

    @staticmethod
    def _serialize_retrieved_docs(result: RAGTestResult) -> list[dict[str, Any]]:
        docs = result.retrieved_contexts
        scores = result.retrieval_scores
        return [
            document_to_record(doc, score=scores[idx] if idx < len(scores) else None)
            for idx, doc in enumerate(docs)
        ]

    def generate_report(
        self,
        *,
        run_dir: Path,
        manifest: dict[str, Any],
        summary_metrics: dict[str, Any],
        ragas_df: pd.DataFrame | None,
        mode: str,
        missed_retrieval_case_ids: list[int],
    ) -> None:
        """Generate human-readable report.md from manifest and metrics."""
        report_text = render_validation_report(
            manifest=manifest,
            summary_metrics=summary_metrics,
            ragas_df=ragas_df,
            mode=mode,
            missed_retrieval_case_ids=missed_retrieval_case_ids,
        )
        report_file = run_dir / "report.md"
        report_file.write_text(report_text, encoding="utf-8")
        logger.info("Validation report:\n%s", report_text.rstrip())
        logger.info("✓ Report saved to %s", report_file)

    def run_retriever_evaluation(
        self,
        dataset_file: Path,
        output_dir: Path,
        sample_size: int | None = None,
        k: int | None = None,
    ) -> dict[str, float | int]:
        """Evaluate retrieve + rerank only: no RAGAS and no answer generation."""
        logger.info("Starting retriever-only evaluation...")

        test_cases = self.load_test_cases(dataset_file)
        results = self.run_retriever_pipeline(test_cases, sample_size)
        retrieval_metrics = self.calculate_retrieval_metrics(results, k=k)
        resolved_k = int(retrieval_metrics.get("k", settings.rag_config.retrieval.dense_top_k))

        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir, run_id, timestamp_iso = build_run_directory(
            output_dir=output_dir, mode="retriever"
        )
        manifest = create_run_manifest(
            mode="retriever",
            run_id=run_id,
            timestamp_iso=timestamp_iso,
            output_dir=output_dir,
            run_dir=run_dir,
            dataset_file=dataset_file,
            sample_size=sample_size,
            k=resolved_k,
            ragas_metric_names=[metric.name for metric in self.ragas_metrics],
        )
        write_json(run_dir / "run_manifest.json", manifest)

        retrieval_cases_file = run_dir / "retrieval_cases.jsonl"
        errors_file = run_dir / "errors.jsonl"
        for result in results:
            gt_norm = result.ground_truth_context.lower().strip()
            matched_rank = None
            for idx, doc in enumerate(result.retrieved_contexts[:resolved_k], start=1):
                if self.contexts_match(gt_norm, doc.page_content.lower().strip()):
                    matched_rank = idx
                    break
            append_jsonl(
                retrieval_cases_file,
                {
                    "test_case_id": result.test_case_id,
                    "question": result.question,
                    "question_type": result.question_type,
                    "section_type": result.section_type,
                    "ground_truth_context": result.ground_truth_context,
                    "k": resolved_k,
                    "matched": matched_rank is not None,
                    "matched_rank": matched_rank,
                    "retrieved_docs": self._serialize_retrieved_docs(result),
                },
            )

        summary_metrics = build_summary_metrics(
            mode="retriever",
            total_loaded_cases=len(test_cases),
            evaluated_cases=len(results),
            rag_success_cases=len(results),
            retrieval_metrics=retrieval_metrics,
            ragas_df=None,
            ragas_metric_names=[metric.name for metric in self.ragas_metrics],
            rag_errors=max(len(test_cases) - len(results), 0),
            ragas_errors=0,
        )
        write_json(run_dir / "summary_metrics.json", summary_metrics)

        missed_retrieval_ids = collect_missed_retrieval_case_ids(
            results=results,
            k=resolved_k,
            matcher=self.contexts_match,
        )
        self.generate_report(
            run_dir=run_dir,
            manifest=manifest,
            summary_metrics=summary_metrics,
            ragas_df=None,
            mode="retriever",
            missed_retrieval_case_ids=missed_retrieval_ids,
        )

        if not errors_file.exists():
            append_jsonl(
                errors_file,
                {
                    "note": "no_errors",
                },
            )
            errors_file.unlink()

        logger.info("✓ Retriever evaluation completed")
        return retrieval_metrics

    def run_full_evaluation(
        self,
        dataset_file: Path,
        output_dir: Path,
        sample_size: int | None = None,
        k: int = 5,
    ) -> tuple[pd.DataFrame, dict[str, float | int]]:
        """Full cycle of RAG system evaluation"""
        logger.info("Launch of a full RAG assessment cycle...")

        # Loading test cases
        test_cases = self.load_test_cases(dataset_file)
        if sample_size and sample_size < len(test_cases):
            test_cases = random.sample(test_cases, sample_size)
            logger.info("A sample of %s cases is used", sample_size)

        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir, run_id, timestamp_iso = build_run_directory(output_dir=output_dir, mode="full")
        manifest = create_run_manifest(
            mode="full",
            run_id=run_id,
            timestamp_iso=timestamp_iso,
            output_dir=output_dir,
            run_dir=run_dir,
            dataset_file=dataset_file,
            sample_size=sample_size,
            k=k,
            ragas_metric_names=[metric.name for metric in self.ragas_metrics],
        )
        write_json(run_dir / "run_manifest.json", manifest)
        cases_file = run_dir / "cases.jsonl"
        errors_file = run_dir / "errors.jsonl"
        ragas_parquet_file = run_dir / "ragas_scores.parquet"

        ragas_errors: list[dict[str, str | int]] = []
        rag_errors: list[dict[str, str | int]] = []
        results: list[RAGTestResult] = []
        ragas_rows: list[pd.DataFrame] = []
        logger.info("Running streaming full evaluation for %s cases...", len(test_cases))

        for idx, test_case in enumerate(tqdm(test_cases, desc="RAG + RAGAS")):
            try:
                rag_result = self.rag_pipeline.run(test_case.question)
            except Exception as e:
                logger.error("Case processing error %s (RAG): %s", idx, e)
                rag_error = {
                    "test_case_id": idx,
                    "stage": "rag",
                    "question": test_case.question,
                    "question_type": test_case.question_type,
                    "section_type": test_case.section_type,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                rag_errors.append(rag_error)
                append_jsonl(errors_file, rag_error)
                append_jsonl(
                    cases_file,
                    {
                        "test_case_id": idx,
                        "status": "rag_error",
                        "question": test_case.question,
                        "ground_truth_answer": test_case.answer,
                        "ground_truth_context": test_case.context,
                        "question_type": test_case.question_type,
                        "section_type": test_case.section_type,
                        "generated_answer": "",
                        "retrieved_docs": [],
                        "ragas_scores": {},
                    },
                )
                continue

            result = RAGTestResult(
                question=test_case.question,
                ground_truth_answer=test_case.answer,
                ground_truth_context=test_case.context,
                retrieved_contexts=rag_result["retrieved_docs"],
                retrieval_scores=rag_result["retrieved_scores"],
                generated_answer=rag_result["answer"],
                question_type=test_case.question_type,
                section_type=test_case.section_type,
                test_case_id=idx,
            )
            results.append(result)
            case_row: dict[str, Any] = {
                "test_case_id": idx,
                "status": "ok",
                "question": result.question,
                "ground_truth_answer": result.ground_truth_answer,
                "ground_truth_context": result.ground_truth_context,
                "generated_answer": result.generated_answer,
                "question_type": result.question_type,
                "section_type": result.section_type,
                "retrieved_docs": self._serialize_retrieved_docs(result),
                "ragas_scores": {},
            }

            try:
                case_ragas_df = self.evaluate_with_ragas([result])
                ragas_rows.append(case_ragas_df)
                case_row["ragas_scores"] = self._extract_numeric_ragas_scores(case_ragas_df.iloc[0])
            except Exception as e:
                logger.error("Case processing error %s (RAGAS): %s", idx, e)
                case_row["status"] = "ragas_error"
                ragas_errors.append(
                    {
                        "test_case_id": idx,
                        "stage": "ragas",
                        "question": test_case.question,
                        "question_type": test_case.question_type,
                        "section_type": test_case.section_type,
                        "generated_answer": result.generated_answer,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                )
                append_jsonl(errors_file, ragas_errors[-1])
            finally:
                append_jsonl(cases_file, case_row)

        ragas_df = (
            pd.concat(ragas_rows, ignore_index=True)
            if ragas_rows
            else pd.DataFrame(columns=["question_type", "section_type", "test_case_id"])
        )
        ragas_df.to_parquet(ragas_parquet_file, index=False)

        # Retriever metrics
        retrieval_metrics = self.calculate_retrieval_metrics(results, k=k)
        missed_retrieval_ids = collect_missed_retrieval_case_ids(
            results=results,
            k=k,
            matcher=self.contexts_match,
        )

        summary_metrics = build_summary_metrics(
            mode="full",
            total_loaded_cases=len(test_cases),
            evaluated_cases=len(results),
            rag_success_cases=len(results),
            retrieval_metrics=retrieval_metrics,
            ragas_df=ragas_df,
            ragas_metric_names=[metric.name for metric in self.ragas_metrics],
            rag_errors=len(rag_errors),
            ragas_errors=len(ragas_errors),
        )
        write_json(run_dir / "summary_metrics.json", summary_metrics)

        # Generating a report
        self.generate_report(
            run_dir=run_dir,
            manifest=manifest,
            summary_metrics=summary_metrics,
            ragas_df=ragas_df,
            mode="full",
            missed_retrieval_case_ids=missed_retrieval_ids,
        )

        logger.info("✓ The full evaluation cycle has been completed")

        return ragas_df, retrieval_metrics

    @staticmethod
    def contexts_match(gt_norm: str, retrieved_norm: str) -> bool:
        gt_tokens = set(re.findall(r"\w+", gt_norm.lower()))
        retrieved_tokens = set(re.findall(r"\w+", retrieved_norm.lower()))
        if not gt_tokens or not retrieved_tokens:
            return False

        overlap = len(gt_tokens & retrieved_tokens)
        gt_coverage = overlap / len(gt_tokens)
        retrieved_coverage = overlap / len(retrieved_tokens)
        jaccard = overlap / len(gt_tokens | retrieved_tokens)

        if overlap < 5:
            return False
        return (gt_coverage >= 0.55 and retrieved_coverage >= 0.35) or jaccard >= 0.4
