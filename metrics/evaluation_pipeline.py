import json
import logging
import random
import re
import statistics
from pathlib import Path

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
from metrics.schemas import QATestCase, RAGTestResult

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

    def generate_report(
        self,
        ragas_df: pd.DataFrame,
        retrieval_metrics: dict[str, float | int],
        output_dir: Path,
    ) -> None:
        """Generating a detailed report"""
        output_dir.mkdir(parents=True, exist_ok=True)

        report_lines = []
        report_lines.append("# RAG SYSTEM TEST REPORT\n\n")

        # 1. General statistics
        report_lines.append("## 1. GENERAL STATISTICS\n")
        report_lines.append(f"Total test cases: {len(ragas_df)}\n")

        # 2. Retriever metrics
        report_lines.append("## 2. RETRIEVER METRICS")
        rk = retrieval_metrics.get("k", settings.rag_config.reranker.top_k)
        report_lines.append(f"Hit Rate (top-{rk}): {retrieval_metrics['hit_rate']:.3f}\n")
        report_lines.append(f"MRR (Mean Reciprocal Rank): {retrieval_metrics['mrr']:.3f}\n")
        report_lines.append(f"Recall@{rk}: {retrieval_metrics['recall_at_k']:.3f}\n")
        report_lines.append(f"Precision@{rk}: {retrieval_metrics['precision_at_k']:.3f}\n")
        report_lines.append(f"Average score top-1: {retrieval_metrics['avg_score']:.3f}\n\n")

        # 3. RAGAS metrics
        report_lines.append("## 3. RAGAS МЕТРИКИ\n")

        for metric in self.ragas_metrics:
            metric_name = metric.name
            if metric_name in ragas_df.columns:
                scores = ragas_df[metric_name].dropna()
                report_lines.append(f"### {metric_name}\n")
                report_lines.append(f"  Mean:  {scores.mean():.3f}\n")
                report_lines.append(f"  Median:  {scores.median():.3f}\n")
                report_lines.append(f"  Std Dev:  {scores.std():.3f}\n")
                report_lines.append(f"  Min-Max: {scores.min():.3f} - {scores.max():.3f}\n\n")

        # 4. Analysis by question types
        report_lines.append("## 4. ANALYSIS BY QUESTION TYPES\n")

        if "question_type" in ragas_df.columns:
            for qtype in ragas_df["question_type"].dropna().unique():
                subset = ragas_df[ragas_df["question_type"] == qtype]
                report_lines.append(f"### {qtype} (n={len(subset)})\n")

                if "faithfulness" in subset.columns:
                    report_lines.append(f"  Faithfulness: {subset['faithfulness'].mean():.3f}\n")
                if "context_recall" in subset.columns:
                    report_lines.append(
                        f"  Context Recall: {subset['context_recall'].mean():.3f}\n"
                    )
                if "answer_correctness" in subset.columns:
                    report_lines.append(
                        f"  Answer Correctness: {subset['answer_correctness'].mean():.3f}\n\n"
                    )
        else:
            report_lines.append("No question_type data available\n\n")

        # 5. Analysis by section types
        report_lines.append("## 5. ANALYSIS BY SECTION TYPES")

        if "section_type" in ragas_df.columns:
            for stype in ragas_df["section_type"].dropna().unique():
                subset = ragas_df[ragas_df["section_type"] == stype]
                report_lines.append(f"### {stype} (n={len(subset)})")

                if "faithfulness" in subset.columns:
                    report_lines.append(f"  Faithfulness: {subset['faithfulness'].mean():.3f}\n")
                if "context_recall" in subset.columns:
                    report_lines.append(
                        f"  Context Recall: {subset['context_recall'].mean():.3f}\n"
                    )
                if "answer_correctness" in subset.columns:
                    report_lines.append(
                        f"  Answer Correctness: {subset['answer_correctness'].mean():.3f}\n\n"
                    )
        else:
            report_lines.append("\nNo section_type data available\n\n")

        # 6. Problematic cases
        report_lines.append("## 6. PROBLEM CASES")

        if "faithfulness" in ragas_df.columns:
            low_faithfulness = ragas_df[ragas_df["faithfulness"] < 0.5]
            report_lines.append(f"Low faithfulness (< 0.5): {len(low_faithfulness)} cases\n")

        if "context_recall" in ragas_df.columns:
            low_recall = ragas_df[ragas_df["context_recall"] < 0.5]
            report_lines.append(f"Low context recall: {len(low_recall)} cases\n")

        if "answer_correctness" in ragas_df.columns:
            low_correctness = ragas_df[ragas_df["answer_correctness"] < 0.4]
            report_lines.append(f"Low answer correctness (< 0.4): {len(low_correctness)} cases\n\n")

        # Save report
        report_text = "".join(report_lines)
        report_file = output_dir / "rag_evaluation_report.md"

        with Path(report_file).open("w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("RAG evaluation report:\n%s", report_text.rstrip())
        logger.info(f"✓ Report saved to {report_file}")

        # Saving detailed results
        csv_file = output_dir / "rag_evaluation_detailed.csv"
        ragas_df.to_csv(csv_file, index=False)
        logger.info(f"✓ Detailed results are saved to {csv_file}")

    def generate_retrieval_report(
        self,
        retrieval_metrics: dict,
        output_dir: Path,
        n_cases: int,
    ) -> None:
        """Persist retriever-only metrics (no RAGAS / no generated answers)."""
        output_dir.mkdir(parents=True, exist_ok=True)

        rk = retrieval_metrics.get("k", settings.rag_config.retrieval.dense_top_k)
        report_lines = [
            "# RETRIEVER EVALUATION REPORT\n\n",
            "## DATASET\n",
            f"Test cases evaluated: {n_cases}\n\n",
            "## RETRIEVER METRICS\n",
            f"K (recall/precision): {rk}\n",
            f"Hit rate: {retrieval_metrics['hit_rate']:.3f}\n",
            f"MRR: {retrieval_metrics['mrr']:.3f}\n",
            f"Recall@{rk}: {retrieval_metrics['recall_at_k']:.3f}\n",
            f"Precision@{rk}: {retrieval_metrics['precision_at_k']:.3f}\n",
            f"Average score (top-1): {retrieval_metrics['avg_score']:.3f}\n",
        ]
        report_text = "".join(report_lines)
        report_file = output_dir / "retriever_evaluation_report.md"
        report_file.write_text(report_text, encoding="utf-8")
        logger.info("Retriever evaluation report:\n%s", report_text.rstrip())
        logger.info("✓ Report saved to %s", report_file)

        metrics_path = output_dir / "retriever_metrics.json"
        metrics_path.write_text(json.dumps(retrieval_metrics, indent=2), encoding="utf-8")
        logger.info("✓ Metrics saved to %s", metrics_path)

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
        self.generate_retrieval_report(retrieval_metrics, output_dir, len(results))

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

        # Launching the RAG pipeline
        results = self.run_rag_pipeline(test_cases, sample_size)

        # RAGAS evaluation
        ragas_df = self.evaluate_with_ragas(results)

        # Retriever metrics
        retrieval_metrics = self.calculate_retrieval_metrics(results, k=k)

        # Generating a report
        self.generate_report(ragas_df, retrieval_metrics, output_dir)

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
