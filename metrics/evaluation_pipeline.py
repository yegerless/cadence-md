import json
import logging
import random
import re
import statistics
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_core.documents import Document
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

from cadence_md.app.settings import settings
from cadence_md.rag import RAGRequest, RAGResponse, RAGRetrieveResponse, RAGService
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _response_to_test_result_inputs(
    response: RAGResponse | RAGRetrieveResponse,
) -> tuple[list[Document], list[float], str]:
    """
    Build aligned ``(documents, scores, answer)`` from stable RAG service DTO response.

    Both lists are derived from ``response.sources`` so document order matches scores
    index-by-index for downstream :class:`RAGTestResult`.

    Args:
        response: Full or retriever-only response from :class:`RAGService`.

    Returns:
        Tuple of (documents, scores, answer) aligned by index.
    """
    docs: list[Document] = []
    scores: list[float] = []
    for source in response.sources:
        docs.append(
            Document(
                page_content=source.content or "",
                metadata={
                    "doc_ref": source.doc_ref,
                    "rank": source.rank,
                    "filename": source.filename,
                    "source_path": source.source_path,
                    "document_title": source.document_title,
                    "section_title": source.section_title,
                    "section_id": source.section_id,
                    "chunk_id": source.chunk_id,
                },
            )
        )
        source_score = source.score
        if source_score is None:
            source_score = source.rerank_score
        if source_score is None:
            source_score = source.retrieval_score
        scores.append(float(source_score) if source_score is not None else 0.0)

    answer = response.answer if isinstance(response, RAGResponse) else ""
    return docs, scores, answer


class RAGEvaluationPipeline:
    """
    Evaluation pipeline for RAG

    Args:
        rag_service: RAG service
        gigachat_llm: GigaChat LLM
        gigachat_embeddings: GigaChat embeddings
        ragas_metrics: List of RAGAS metrics to evaluate
    Returns:
        Evaluation pipeline for RAG
    """

    def __init__(
        self,
        rag_service: RAGService,
        gigachat_llm: GigaChat,
        gigachat_embeddings: GigaChatEmbeddings,
        ragas_metrics: list | None = None,
        rag_optional_nodes_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the evaluation pipeline for RAG

        Args:
            rag_service: RAG service
            gigachat_llm: GigaChat LLM (used for evaluation by api calls inside RAGAS)
            gigachat_embeddings: GigaChat embeddings (used for evaluation by api calls inside RAGAS)
            ragas_metrics: List of RAGAS metrics to evaluate
            rag_optional_nodes_config: Effective optional RAG node settings for artifacts
        Returns:
            Evaluation pipeline for RAG with RAGAS metrics
        """
        self.rag_service = rag_service
        self.rag_optional_nodes_config = rag_optional_nodes_config

        # Initialize evaluation LLM and embeddings for ragas metrics
        self.evaluation_llm = LangchainLLMWrapper(gigachat_llm)
        self.evaluation_embeddings = LangchainEmbeddingsWrapper(gigachat_embeddings)

        # If no RAGAS metrics are provided, use the default metrics
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
            timeout=300,  # 5 minutes
            max_retries=15,
            max_wait=120,  # 2 minutes
            log_tenacity=True,
        )

        logger.info(f"Initialized evaluation pipeline with {len(self.ragas_metrics)} metrics")

    def load_test_cases(self, dataset_file: Path) -> list[QATestCase]:
        """
        Loading test cases from JSONL

        Args:
            dataset_file: Path to the dataset file
        Returns:
            List of test cases
        """
        test_cases: list[QATestCase] = []

        with Path(dataset_file).open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        test_cases.append(QATestCase.from_dict(data))
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        logger.warning(
                            f"Skipping malformed QA row at {dataset_file}:{line_num} ({exc})"
                        )

        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases

    def run_rag_pipeline(
        self,
        test_cases: list[QATestCase],
        sample_size: int | None = None,
    ) -> list[RAGTestResult]:
        """
        Running the RAG pipeline for all test cases

        Args:
            test_cases: List of test cases
            sample_size: Number of test cases to sample
        Returns:
            List of RAG test results
        """

        # If sample size is provided, sample the test cases
        if sample_size and sample_size < len(test_cases):
            test_cases = random.sample(test_cases, sample_size)
            logger.info(f"A sample of {sample_size} cases is used")

        results: list[RAGTestResult] = []

        logger.info(f"Running the RAG pipeline for {len(test_cases)} cases...")

        # Run the RAG pipeline for all test cases
        for idx, test_case in enumerate(tqdm(test_cases, desc="RAG inference")):
            try:
                rag_result = self.rag_service.run(
                    RAGRequest(query=test_case.question, allow_clarification=False)
                )
                retrieved_contexts, retrieved_scores, generated_answer = (
                    _response_to_test_result_inputs(rag_result)
                )

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
                    ground_truth_section_id=test_case.section_id,
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
        workers: int = 1,
    ) -> list[RAGTestResult]:
        """
        Retrieve + rerank only; no LLM generation (retriever-focused evaluation).

        Args:
            test_cases: List of test cases
            sample_size: Number of test cases to sample
            workers: Number of parallel worker threads for independent retriever cases
        Returns:
            List of RAG test results
        """
        if workers <= 0:
            raise ValueError("workers must be a positive integer")

        # If sample size is provided, sample the test cases
        if sample_size and sample_size < len(test_cases):
            test_cases = random.sample(test_cases, sample_size)
            logger.info(f"A sample of {sample_size} cases is used")

        logger.info(
            "Running retriever (retrieve + rerank) for %s cases with %s worker(s)...",
            len(test_cases),
            workers,
        )

        indexed_cases = list(enumerate(test_cases))
        if workers == 1:
            results = [
                result
                for idx, test_case in tqdm(indexed_cases, desc="Retriever")
                if (result := self._run_retriever_case(idx, test_case)) is not None
            ]
        else:
            results_by_idx: dict[int, RAGTestResult] = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._run_retriever_case, idx, test_case): idx
                    for idx, test_case in indexed_cases
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Retriever",
                ):
                    idx = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.error("Case processing error %s: %s", idx, exc)
                        continue
                    if result is not None:
                        results_by_idx[idx] = result
            results = [results_by_idx[idx] for idx in sorted(results_by_idx)]

        logger.info(f"✓ Processed {len(results)} cases (retriever only)")
        return results

    def _run_retriever_case(
        self,
        idx: int,
        test_case: QATestCase,
    ) -> RAGTestResult | None:
        """
        Run retrieve + rerank for a single QA case.

        Args:
            idx: Stable case id within the sampled evaluation set
            test_case: QA test case
        Returns:
            RAG test result, or None when the case fails
        """
        try:
            retrieve_result = self.rag_service.retrieve(
                RAGRequest(query=test_case.question, allow_clarification=False)
            )
            retrieved_contexts, retrieval_scores, _ = _response_to_test_result_inputs(
                retrieve_result
            )

            return RAGTestResult(
                question=test_case.question,
                ground_truth_answer=test_case.answer,
                ground_truth_context=test_case.context,
                retrieved_contexts=retrieved_contexts,
                retrieval_scores=retrieval_scores,
                generated_answer="",
                question_type=test_case.question_type,
                section_type=test_case.section_type,
                test_case_id=idx,
                ground_truth_section_id=test_case.section_id,
                retrieval_query=retrieve_result.retrieval_query,
                rewritten_queries=list(retrieve_result.rewritten_queries),
                rag_flags=retrieve_result.flags.model_dump(),
                context_relevance_score=retrieve_result.context_relevance_score,
            )

        except Exception as e:
            logger.error(f"Case processing error {idx}: {e}")
            return None

    def convert_to_ragas_format(self, results: list[RAGTestResult]) -> Dataset:
        """
        Converting results to RAGAS format

        Args:
            results: List of RAG test results
        Returns:
            Dataset in RAGAS format
        """
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
        """
        Assessing results using RAGAS metrics (legacy evaluate API)

        Args:
            results: List of RAG test results
        Returns:
            DataFrame with RAGAS scores
        """

        # If no results are provided, return an empty DataFrame
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
        """
        Extract only numeric RAGAS scores from a mixed row

        Args:
            ragas_row: Series with RAGAS scores
        Returns:
            Dictionary with numeric RAGAS scores
        """
        # Define columns that are not numeric scores
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
        matcher: Callable[[RAGTestResult, Document], bool] | None = None,
        metric_prefix: str = "",
    ) -> dict[str, float | int]:
        """
        Calculation of retriever metrics

        Args:
            results: List of RAG test results
            k: Number of retrieved documents to evaluate
            matcher: Matcher to use for evaluation
            metric_prefix: Prefix for the metric keys
        Returns:
            Dictionary with retrieval metrics
        """

        # If k is not provided, use the default value from settings
        k = k if k is not None else settings.rag_config.reranker.top_k

        if k <= 0:
            raise ValueError("k must be a positive integer")

        k_key = f"{metric_prefix}k"
        metrics: dict[str, float | int] = {
            f"{metric_prefix}hit_rate": 0.0,
            f"{metric_prefix}mrr": 0.0,
            f"{metric_prefix}avg_score": 0.0,
            f"{metric_prefix}recall_at_k": 0.0,
            f"{metric_prefix}precision_at_k": 0.0,
            k_key: k,
        }

        if not results:
            return metrics

        relevance_matcher = matcher or self.section_ids_match
        hits = 0
        reciprocal_ranks: list[float] = []
        top_1_scores: list[float] = []
        recall_at_k_values: list[float] = []
        precision_at_k_values: list[float] = []

        # Calculate the metrics for each result
        for result in results:
            found = False
            # Check if the retrieved context matches the relevance matcher
            for i, retrieved_ctx in enumerate(result.retrieved_contexts):
                if relevance_matcher(result, retrieved_ctx):
                    hits += 1
                    reciprocal_ranks.append(1.0 / (i + 1))
                    found = True
                    break

            if not found:
                reciprocal_ranks.append(0.0)

            if result.retrieval_scores:
                top_1_scores.append(result.retrieval_scores[0])

            # Get the top k retrieved documents
            top_k_docs = result.retrieved_contexts[:k]
            relevant_in_top_k = sum(1 for doc in top_k_docs if relevance_matcher(result, doc))
            recall_at_k_values.append(1.0 if relevant_in_top_k > 0 else 0.0)
            precision_at_k_values.append((relevant_in_top_k / k) if k > 0 else 0.0)

        # Calculate the metrics
        metrics[f"{metric_prefix}hit_rate"] = hits / len(results)
        metrics[f"{metric_prefix}mrr"] = statistics.mean(reciprocal_ranks)
        metrics[f"{metric_prefix}avg_score"] = (
            statistics.mean(top_1_scores) if top_1_scores else 0.0
        )
        metrics[f"{metric_prefix}recall_at_k"] = statistics.mean(recall_at_k_values)
        metrics[f"{metric_prefix}precision_at_k"] = statistics.mean(precision_at_k_values)

        return metrics

    @staticmethod
    def section_ids_match(result: RAGTestResult, retrieved_doc: Document) -> bool:
        """
        Return True when retrieved chunk belongs to the expected source section

        Args:
            result: RAG test result
            retrieved_doc: Retrieved document
        Returns:
            True if the retrieved document belongs to the expected source section
        """
        expected_section_id = result.ground_truth_section_id.strip()
        retrieved_section_id = str(retrieved_doc.metadata.get("section_id", "")).strip()
        return bool(expected_section_id and retrieved_section_id == expected_section_id)

    def text_matcher_matches(self, result: RAGTestResult, retrieved_doc: Document) -> bool:
        """
        Return True when retrieved text matches the reference evidence text

        Args:
            result: RAG test result
            retrieved_doc: Retrieved document
        Returns:
            True if the retrieved document matches the reference evidence text
        """
        return self.contexts_match(result.ground_truth_context, retrieved_doc.page_content)

    @staticmethod
    def _matched_rank(
        result: RAGTestResult,
        *,
        k: int,
        matcher: Callable[[RAGTestResult, Document], bool],
    ) -> int | None:
        """
        Return the rank of the first retrieved document that matches the matcher

        Args:
            result: RAG test result
            k: Number of retrieved documents to evaluate
            matcher: Matcher to use for evaluation
        Returns:
            Rank of the first retrieved document that matches the matcher
        """
        for idx, doc in enumerate(result.retrieved_contexts[:k], start=1):
            if matcher(result, doc):
                return idx
        return None

    @staticmethod
    def _serialize_retrieved_docs(result: RAGTestResult) -> list[dict[str, Any]]:
        """
        Serialize the retrieved documents and scores

        Args:
            result: RAG test result
        Returns:
            List of dictionaries with the retrieved documents and scores
        """
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
        """
        Generate human-readable report.md from manifest and metrics

        Args:
            run_dir: Path to the run directory
            manifest: Manifest for the run
            summary_metrics: Summary metrics for the run
            ragas_df: RAGAS dataframe for the run
            mode: Mode of the run
            missed_retrieval_case_ids: List of case ids that were missed in retrieval
        Returns:
            None
        """
        report_text = render_validation_report(
            manifest=manifest,
            summary_metrics=summary_metrics,
            ragas_df=ragas_df,
            mode=mode,
            missed_retrieval_case_ids=missed_retrieval_case_ids,
        )
        report_file = run_dir / "report.md"
        report_file.write_text(report_text, encoding="utf-8")
        logger.info(f"Validation report:\n{report_text.rstrip()}")
        logger.info(f"Report saved to {report_file}")

    def run_retriever_evaluation(
        self,
        dataset_file: Path,
        output_dir: Path,
        sample_size: int | None = None,
        k: int | None = None,
        enable_text_matcher_metrics: bool = False,
        workers: int = 1,
    ) -> dict[str, float | int]:
        """
        Evaluate retrieve + rerank only: no RAGAS and no answer generation

        Args:
            dataset_file: Path to the dataset file
            output_dir: Path to the output directory
            sample_size: Number of test cases to sample
            k: Number of retrieved documents to evaluate
            enable_text_matcher_metrics: Whether to enable text matcher metrics
            workers: Number of parallel worker threads for independent retriever cases
        Returns:
            Dictionary with retrieval metrics
        """
        logger.info("Starting retriever-only evaluation...")

        test_cases = self.load_test_cases(dataset_file)
        results = self.run_retriever_pipeline(test_cases, sample_size, workers=workers)
        retrieval_metrics = self.calculate_retrieval_metrics(results, k=k)
        resolved_k = int(retrieval_metrics.get("k", settings.rag_config.retrieval.dense_top_k))
        text_match_retrieval_metrics = (
            self.calculate_retrieval_metrics(
                results,
                k=resolved_k,
                matcher=self.text_matcher_matches,
                metric_prefix="text_match_",
            )
            if enable_text_matcher_metrics
            else None
        )

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
            enable_text_matcher_metrics=enable_text_matcher_metrics,
            workers=workers,
            rag_optional_nodes_config=getattr(self, "rag_optional_nodes_config", None),
        )
        write_json(run_dir / "run_manifest.json", manifest)

        retrieval_cases_file = run_dir / "retrieval_cases.jsonl"
        errors_file = run_dir / "errors.jsonl"
        for result in results:
            matched_rank = self._matched_rank(
                result,
                k=resolved_k,
                matcher=self.section_ids_match,
            )
            retrieval_case = {
                "test_case_id": result.test_case_id,
                "question": result.question,
                "question_type": result.question_type,
                "section_type": result.section_type,
                "ground_truth_context": result.ground_truth_context,
                "ground_truth_section_id": result.ground_truth_section_id,
                "k": resolved_k,
                "matched": matched_rank is not None,
                "matched_rank": matched_rank,
                "retrieval_query": result.retrieval_query,
                "rewritten_queries": result.rewritten_queries,
                "rag_flags": result.rag_flags,
                "context_relevance_score": result.context_relevance_score,
                "retrieved_docs": self._serialize_retrieved_docs(result),
            }
            if enable_text_matcher_metrics:
                text_match_rank = self._matched_rank(
                    result,
                    k=resolved_k,
                    matcher=self.text_matcher_matches,
                )
                retrieval_case["text_match_matched"] = text_match_rank is not None
                retrieval_case["text_match_matched_rank"] = text_match_rank
            append_jsonl(
                retrieval_cases_file,
                retrieval_case,
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
            text_match_retrieval_metrics=text_match_retrieval_metrics,
        )
        write_json(run_dir / "summary_metrics.json", summary_metrics)

        missed_retrieval_ids = collect_missed_retrieval_case_ids(
            results=results,
            k=resolved_k,
            matcher=self.section_ids_match,
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

        logger.info("✓ Retriever evaluation completed successfully")
        return retrieval_metrics

    def run_full_evaluation(
        self,
        dataset_file: Path,
        output_dir: Path,
        sample_size: int | None = None,
        k: int = 5,
        enable_text_matcher_metrics: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, float | int]]:
        """
        Full cycle of RAG system evaluation

        Args:
            dataset_file: Path to the dataset file
            output_dir: Path to the output directory
            sample_size: Number of test cases to sample
            k: Number of retrieved documents to evaluate
            enable_text_matcher_metrics: Whether to enable text matcher metrics
        Returns:
            Tuple containing the RAGAS dataframe and retrieval metrics
        """

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
            enable_text_matcher_metrics=enable_text_matcher_metrics,
            rag_optional_nodes_config=getattr(self, "rag_optional_nodes_config", None),
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
                rag_result = self.rag_service.run(
                    RAGRequest(query=test_case.question, allow_clarification=False)
                )
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
                        "ground_truth_section_id": test_case.section_id,
                        "question_type": test_case.question_type,
                        "section_type": test_case.section_type,
                        "generated_answer": "",
                        "retrieved_docs": [],
                        "ragas_scores": {},
                    },
                )
                continue

            retrieved_contexts, retrieval_scores, generated_answer = (
                _response_to_test_result_inputs(rag_result)
            )
            result = RAGTestResult(
                question=test_case.question,
                ground_truth_answer=test_case.answer,
                ground_truth_context=test_case.context,
                retrieved_contexts=retrieved_contexts,
                retrieval_scores=retrieval_scores,
                generated_answer=generated_answer,
                question_type=test_case.question_type,
                section_type=test_case.section_type,
                test_case_id=idx,
                ground_truth_section_id=test_case.section_id,
            )
            results.append(result)
            matched_rank = self._matched_rank(result, k=k, matcher=self.section_ids_match)
            case_row: dict[str, Any] = {
                "test_case_id": idx,
                "status": "ok",
                "question": result.question,
                "ground_truth_answer": result.ground_truth_answer,
                "ground_truth_context": result.ground_truth_context,
                "ground_truth_section_id": result.ground_truth_section_id,
                "generated_answer": result.generated_answer,
                "question_type": result.question_type,
                "section_type": result.section_type,
                "retrieval_matched": matched_rank is not None,
                "retrieval_matched_rank": matched_rank,
                "retrieved_docs": self._serialize_retrieved_docs(result),
                "ragas_scores": {},
            }
            if enable_text_matcher_metrics:
                text_match_rank = self._matched_rank(result, k=k, matcher=self.text_matcher_matches)
                case_row["text_match_matched"] = text_match_rank is not None
                case_row["text_match_matched_rank"] = text_match_rank

            try:
                case_ragas_df = self.evaluate_with_ragas([result])
                ragas_rows.append(case_ragas_df)
                case_row["ragas_scores"] = self._extract_numeric_ragas_scores(case_ragas_df.iloc[0])
            except Exception as e:
                logger.error(f"Case processing error {idx} (RAGAS): {e}")
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
        text_match_retrieval_metrics = (
            self.calculate_retrieval_metrics(
                results,
                k=k,
                matcher=self.text_matcher_matches,
                metric_prefix="text_match_",
            )
            if enable_text_matcher_metrics
            else None
        )
        missed_retrieval_ids = collect_missed_retrieval_case_ids(
            results=results,
            k=k,
            matcher=self.section_ids_match,
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
            text_match_retrieval_metrics=text_match_retrieval_metrics,
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

        logger.info("The full evaluation cycle has been completed successfully")

        return ragas_df, retrieval_metrics

    @staticmethod
    def contexts_match(gt_norm: str, retrieved_norm: str) -> bool:
        """
        Check if the ground truth context matches the retrieved context

        Args:
            gt_norm: Normalized ground truth context
            retrieved_norm: Normalized retrieved context
        Returns:
            True if the ground truth context matches the retrieved context
        """
        # Normalize the ground truth and retrieved contexts
        gt_text = RAGEvaluationPipeline._normalize_match_text(gt_norm)
        retrieved_text = RAGEvaluationPipeline._normalize_match_text(retrieved_norm)
        if not gt_text or not retrieved_text:
            return False

        # Check if the shorter text is at least 30 characters and contains the other text
        shorter_text = gt_text if len(gt_text) <= len(retrieved_text) else retrieved_text
        if len(shorter_text) >= 30 and (gt_text in retrieved_text or retrieved_text in gt_text):
            return True

        # Check if the ground truth and retrieved contexts have meaningful tokens
        gt_tokens = RAGEvaluationPipeline._meaningful_tokens(gt_text)
        retrieved_tokens = RAGEvaluationPipeline._meaningful_tokens(retrieved_text)
        if not gt_tokens or not retrieved_tokens:
            return False

        # Check if the ground truth and retrieved contexts have meaningful overlap
        overlap = len(gt_tokens & retrieved_tokens)
        gt_coverage = overlap / len(gt_tokens)
        retrieved_coverage = overlap / len(retrieved_tokens)
        jaccard = overlap / len(gt_tokens | retrieved_tokens)

        # Check if the overlap is less than 5
        if overlap < 5:
            return False

        # Check if the ground truth and retrieved contexts have meaningful coverage
        if (gt_coverage >= 0.6 and retrieved_coverage >= 0.25) or jaccard >= 0.35:
            return True

        return RAGEvaluationPipeline._char_ngram_match(
            gt_text=gt_text,
            retrieved_text=retrieved_text,
            gt_coverage=gt_coverage,
        )

    @staticmethod
    def _normalize_match_text(value: str) -> str:
        """
        Normalize the text for matching

        Args:
            value: Text to normalize
        Returns:
            Normalized text
        """
        text = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
        text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _meaningful_tokens(value: str) -> set[str]:
        """
        Extract meaningful tokens from the text

        Args:
            value: Text to extract tokens from
        Returns:
            Set of meaningful tokens
        """
        stopwords = {
            "без",
            "более",
            "быть",
            "для",
            "его",
            "или",
            "как",
            "которые",
            "может",
            "над",
            "при",
            "также",
            "что",
            "это",
        }
        return {
            token
            for token in re.findall(r"\w+", value, flags=re.UNICODE)
            if len(token) > 2 and token not in stopwords
        }

    @staticmethod
    def _char_ngrams(value: str, n: int = 5) -> set[str]:
        """
        Extract character ngrams from the text

        Args:
            value: Text to extract ngrams from
            n: Length of the ngrams
        Returns:
            Set of character ngrams
        """
        compact = re.sub(r"\s+", " ", value)
        if len(compact) < n:
            return set()
        return {compact[idx : idx + n] for idx in range(len(compact) - n + 1)}

    @staticmethod
    def _char_ngram_match(*, gt_text: str, retrieved_text: str, gt_coverage: float) -> bool:
        """
        Check if the ground truth text matches the retrieved text

        Args:
            gt_text: Ground truth text
            retrieved_text: Retrieved text
            gt_coverage: Coverage of the ground truth text
        Returns:
            True if the ground truth text matches the retrieved text
        """
        gt_ngrams = RAGEvaluationPipeline._char_ngrams(gt_text)
        retrieved_ngrams = RAGEvaluationPipeline._char_ngrams(retrieved_text)
        if not gt_ngrams or not retrieved_ngrams:
            return False
        ngram_overlap = len(gt_ngrams & retrieved_ngrams)
        dice = (2 * ngram_overlap) / (len(gt_ngrams) + len(retrieved_ngrams))
        return gt_coverage >= 0.4 and dice >= 0.62
