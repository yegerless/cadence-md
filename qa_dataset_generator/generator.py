import json
import logging
import random
import time
from collections import defaultdict
from http import HTTPStatus
from pathlib import Path
from typing import Any, TypedDict

from gigachat.exceptions import ResponseError
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat.chat_models import GigaChat
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from tqdm import tqdm

from cadence_md.app.enums import SectionType
from cadence_md.app.pdf_parser import ClinicalSection
from qa_dataset_generator.config import (
    CONTEXT_FALLBACK_MAX_LEN,
    DEFAULT_MAX_CONTEXT_LENGTH,
    DEFAULT_MIN_CONTEXT_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_SECTIONS_PER_PDF,
    DEFAULT_TEMPERATURE,
    GENERATION_TQDM_DESC,
    GENERATION_TQDM_UNIT,
    GIGACHAT_API_KEY,
    GIGACHAT_API_KEY_ENV,
    GIGACHAT_VERIFY_SSL_CERTS,
    INITIAL_RETRY_DELAY_SEC,
    INTERMEDIATE_SAVE_SECTION_INTERVAL,
    MAX_LLM_RETRY_ATTEMPTS,
    MAX_RETRY_DELAY_SEC,
    MIN_ANSWER_LENGTH,
    MIN_QUESTION_LENGTH,
    RETRY_JITTER_MAX_FACTOR,
    SECTION_QUESTION_TYPES,
    UNKNOWN_SOURCE_KEY,
)
from qa_dataset_generator.enums import QuestionType
from qa_dataset_generator.prompts import GENERATION_PROMPTS
from qa_dataset_generator.schemas import (
    GenerationPipelineStats,
    QAPair,
    QAResponsePair,
    _SelectionStats,
)

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# langgraph state
class QAGenerationState(TypedDict):
    """State of the LangGraph graph"""

    section: ClinicalSection
    question_type: str
    generated_pair: QAPair | None
    errors: list[str]


def _is_rate_limit_error(error: BaseException) -> bool:
    """Check if exception indicates provider rate limiting."""
    if isinstance(error, ResponseError) and len(error.args) >= 2:
        return int(error.args[1]) == HTTPStatus.TOO_MANY_REQUESTS
    return False


def _is_retryable_error(error: BaseException) -> bool:
    """Check whether the error can be retried."""
    return _is_rate_limit_error(error) or isinstance(error, TimeoutError | ConnectionError)


def _invoke_with_retry(
    chain: Any,
    payload: dict[str, str],
    max_attempts: int = MAX_LLM_RETRY_ATTEMPTS,
) -> dict[str, Any]:
    """Invoke chain with retry/backoff for temporary provider failures."""
    delay = INITIAL_RETRY_DELAY_SEC
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = chain.invoke(payload)
            if not isinstance(result, dict):
                raise ValueError(f"Unexpected result format: {type(result)}")
            return result
        except Exception as error:
            if not _is_retryable_error(error) or attempt == max_attempts:
                raise
            last_error = error
            logger.warning(
                f"Temporary LLM error, retrying in {delay:.1f} sec ({attempt}/{max_attempts}): "
                f"{error}"
            )
            jitter = random.uniform(0, RETRY_JITTER_MAX_FACTOR * delay)
            time.sleep(delay + jitter)
            delay = min(MAX_RETRY_DELAY_SEC, delay * 2)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry loop finished without result or error")


class QAGeneratorNodes:
    """
    Nodes for LangGraph QA generation graph

    Nodes:
    - initialize_generation: Initialization - Defining Question Types
    - generate_questions: Question generation via LLM
    - validate_qa_pair: Validation of QA pairs
    """

    def __init__(self, llm: GigaChat, max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH):
        """
        Initialization of the nodes

        Args:
            llm: LLM model
            max_context_length: Maximum context length
        """
        self.llm = llm
        self.max_context_length = max_context_length
        # Parser for the LLM response
        self.parser = JsonOutputParser(pydantic_object=QAResponsePair)

    def initialize_generation(self, state: QAGenerationState) -> QAGenerationState:
        """
        Node 1 (initialization) - defining question type for the section

        Args:
            state: State of the LangGraph graph with question section
        Returns:
            State of the LangGraph graph with the question type
        """
        section = state["section"]
        section_type = SectionType(section.section_type)

        question_types = SECTION_QUESTION_TYPES.get(section_type, [QuestionType.SIMPLE])

        state["question_type"] = random.choice([qt.value for qt in question_types])
        state["generated_pair"] = None
        state["errors"] = []

        logger.info(
            f"Initialization for section '{section.section_type}', "
            f"question type: {state['question_type']}"
        )

        return state

    def generate_questions(self, state: QAGenerationState) -> QAGenerationState:
        """Node 2 (generation) - question generation via LLM

        Args:
            state: State of the LangGraph graph with question section and question type
        Returns:
            State of the LangGraph graph with the generated QA pair
        """
        section = state["section"]
        question_type = state["question_type"]

        if not question_type:
            return state

        # Get prompt
        prompt_key = (section.section_type, question_type)
        prompt_template = GENERATION_PROMPTS.get(prompt_key)

        if not prompt_template:
            error_msg = f"Prompt not found for {prompt_key}"
            logger.warning(error_msg)
            state["errors"].append(error_msg)
            return state

        # Trim context if necessary.
        context = section.content[: self.max_context_length]

        # Forming a prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Chain: prompt -> LLM -> JSON parser
        chain = prompt | self.llm | self.parser

        try:
            # Call LLM
            result = _invoke_with_retry(chain=chain, payload={"context": context})

            # Convert result to QAPair
            pair = QAPair(
                question=result.get("question", ""),
                answer=result.get("answer", ""),
                question_type=question_type,
                section_type=section.section_type,
                document_title=section.document_title,
                section_title=section.section_title,
                mkb_codes=section.mkb_codes,
                context=result.get("context", context[:CONTEXT_FALLBACK_MAX_LEN]),
                section_id=section.section_id,
            )
            state["generated_pair"] = pair

            logger.info("QA pair generated successfully")

        except Exception as e:
            error_msg = f"Generation error: {e}"
            logger.error(error_msg)
            state["errors"].append(error_msg)

        return state

    def validate_qa_pair(self, state: QAGenerationState) -> QAGenerationState:
        """Node 3 (validation) - validation of QA pairs

        Args:
            state: State of the LangGraph graph with the generated QA pair
        Returns:
            State of the LangGraph graph with the validated QA pair or None if the pair is invalid
        """
        invalid = False
        pair = state["generated_pair"]

        # Base checks
        if not pair:
            state["errors"].append("generated_pair not found")
            invalid = True
        elif len(pair.question) < MIN_QUESTION_LENGTH:
            state["errors"].append(f"Question too short: {pair.question}")
            state["generated_pair"] = None
        elif len(pair.answer) < MIN_ANSWER_LENGTH:
            state["errors"].append(f"Answer too short: {pair.question}")
            state["generated_pair"] = None

        # Direct copy check
        elif pair.question.lower() in pair.context.lower():
            state["errors"].append(f"Question copy context: {pair.question}")
            state["generated_pair"] = None

        if invalid:
            state["generated_pair"] = None
            logger.warning("Filtered invalid QA pair")

        return state


class QAGenerationGraph:
    """
    LangGraph graph for generating QA datasets

    Nodes:
    - initialize: Initialization - defining question type for the section
    - generate: Question generation via LLM
    - validate: Validation of QA pairs

    Edges:
    - START -> initialize
    - initialize -> generate
    - generate -> validate
    - validate -> END
    """

    def __init__(self, llm: GigaChat, max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH):
        """
        Initialization of the graph

        Args:
            llm: LLM model
            max_context_length: Maximum context length
        """
        self.nodes = QAGeneratorNodes(llm, max_context_length)
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """
        Construction of a graph

        Returns:
            CompiledStateGraph: Compiled graph
        """
        workflow = StateGraph(QAGenerationState)

        # Add nodes
        workflow.add_node("initialize", self.nodes.initialize_generation)
        workflow.add_node("generate", self.nodes.generate_questions)
        workflow.add_node("validate", self.nodes.validate_qa_pair)

        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", END)

        return workflow.compile()

    def run(self, section: ClinicalSection) -> QAPair | None:
        """
        Running a graph for one section

        Args:
            section: Section to generate questions for
        Returns:
            QAPair: Generated QA pair or None if the generation failed
        """
        logger.info(f"Start generation for: {section.document_title} / {section.section_title}")

        initial_state: QAGenerationState = {
            "section": section,
            "question_type": "",
            "generated_pair": None,
            "errors": [],
        }

        final_state = self.graph.invoke(initial_state)

        if final_state["generated_pair"]:
            logger.info(
                f"Generation complete successfully: QA pair generated, "
                f"{len(final_state['errors'])} errors"
            )
        else:
            logger.warning(f"Failed generation, {len(final_state['errors'])} errors")

        return final_state["generated_pair"]


class QADatasetGenerator:
    """
    Synthetic dataset generator for all sections
    This class is used to generate a synthetic dataset from a file with sections
    and save it to a JSONL file

    Raises:
        ValueError: If sections_per_pdf is less than or equal to 0
        ValueError: If min_context_length is less than 0
        ValueError: If the output file already exists

    Examples:
        >>> generator = QADatasetGenerator(
        >>>     model_name="GigaChat-2-Max",
        >>>     temperature=0.0,
        >>>     max_context_length=10000,
        >>>     min_context_length=500,
        >>>     sections_per_pdf=3,
        >>>     seed=42,
        >>> )
        >>> generator.generate_from_sections_file(
        >>>     sections_file="data/clinical_sections.jsonl",
        >>>     output_file="data/qa_dataset.jsonl",
        >>> )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
        min_context_length: int = DEFAULT_MIN_CONTEXT_LENGTH,
        sections_per_pdf: int = DEFAULT_SECTIONS_PER_PDF,
        seed: int | None = None,
    ):
        """
        Initialization of the generator

        Args:
            model_name: Name of the LLM model
            temperature: Temperature for the LLM model
            max_context_length: Maximum context length for the LLM model
            min_context_length: Minimum context length for the LLM model
            sections_per_pdf: Number of sections per PDF for the LLM model
            seed: Seed for the random number generator for the LLM model
        """
        if sections_per_pdf <= 0:
            raise ValueError("sections_per_pdf must be greater than 0")
        if min_context_length < 0:
            raise ValueError("min_context_length must be >= 0")
        self.sections_per_pdf = sections_per_pdf
        self.min_context_length = min_context_length
        self.rng = random.Random(seed)

        # init LLM
        llm_kwargs = {
            "model": model_name,
            "verify_ssl_certs": GIGACHAT_VERIFY_SSL_CERTS,
            "temperature": temperature,
        }

        if not GIGACHAT_API_KEY:
            raise ValueError(
                f"{GIGACHAT_API_KEY_ENV} is not set",
            )
        llm_kwargs["credentials"] = GIGACHAT_API_KEY

        self.llm = GigaChat(**llm_kwargs)

        # Build graph
        self.graph = QAGenerationGraph(llm=self.llm, max_context_length=max_context_length)

    def generate_from_sections_file(
        self, sections_file: Path, output_file: Path
    ) -> tuple[list[QAPair], GenerationPipelineStats]:
        """
        Generating QA from a file with sections

        Args:
            sections_file: Path to the file with sections
            output_file: Path to the file to save the generated QA pairs
        Returns:
            tuple[list[QAPair], GenerationPipelineStats]: Generated QA pairs and statistics

        Raises:
            FileExistsError: If the output file already exists
        """
        logger.info(f"Loading sections from {sections_file}")
        if output_file.exists():
            raise FileExistsError(f"Output file already exists: {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load sections
        sections: list[ClinicalSection] = []
        skipped_invalid_jsonl = 0
        with Path(sections_file).open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as error:
                        skipped_invalid_jsonl += 1
                        logger.warning(f"Skipping invalid JSON at line {line_number}: {error}")
                        continue
                    try:
                        sections.append(ClinicalSection.from_dict(data))
                    except (TypeError, ValueError) as error:
                        skipped_invalid_jsonl += 1
                        logger.warning(f"Skipping invalid section at line {line_number}: {error}")
                        continue
        selected_sections, selection_stats = self._select_random_sections(sections)
        logger.info(
            f"Loaded {len(sections)} sections, selected {len(selected_sections)} for generation, "
            f"skipped_invalid_jsonl={skipped_invalid_jsonl}, "
            f"filtered_short={selection_stats.sections_filtered_short}, "
            f"sources_total={selection_stats.sources_total}, "
            f"sources_skipped_short={selection_stats.sources_skipped_short}",
        )

        # Generate QA for selected sections
        all_pairs: list[QAPair] = []
        pending_pairs: list[QAPair] = []
        processed = 0
        failed = 0

        pbar = tqdm(
            enumerate(selected_sections, 1),
            total=len(selected_sections),
            desc=GENERATION_TQDM_DESC,
            unit=GENERATION_TQDM_UNIT,
        )
        for i, section in pbar:
            processed += 1

            try:
                pair = self.graph.run(section)
                if pair is None:
                    failed += 1
                else:
                    all_pairs.append(pair)
                    pending_pairs.append(pair)
            except Exception as e:
                failed += 1
                logger.error(f"Error processing section {section.section_title}: {e}")

            # Intermediate save
            if i % INTERMEDIATE_SAVE_SECTION_INTERVAL == 0 and pending_pairs:
                self._append_pairs(pending_pairs, output_file)
                logger.info(
                    f"Intermediate storage: {len(all_pairs)} total generated, {failed} failed",
                )
                pending_pairs = []

            pbar.set_postfix(generated=len(all_pairs), failed=failed)

        # Final save
        if pending_pairs:
            self._append_pairs(pending_pairs, output_file)
        logger.info(
            f"Generation complete: processed={processed} generated={len(all_pairs)} failed={failed}"
            f" skipped_invalid_jsonl={skipped_invalid_jsonl}"
        )

        stats = GenerationPipelineStats(
            sections_loaded=len(sections),
            sections_selected=len(selected_sections),
            skipped_invalid_jsonl=skipped_invalid_jsonl,
            sections_processed=processed,
            pairs_generated=len(all_pairs),
            sections_failed=failed,
            min_context_length=self.min_context_length,
            sources_total=selection_stats.sources_total,
            sources_skipped_short=selection_stats.sources_skipped_short,
            skipped_sources=tuple(selection_stats.skipped_sources),
            sections_filtered_short=selection_stats.sections_filtered_short,
        )
        return all_pairs, stats

    def _select_random_sections(
        self, sections: list[ClinicalSection]
    ) -> tuple[list[ClinicalSection], _SelectionStats]:
        """
        Select N sections per source with balanced section_type distribution.
        Applies a minimum context length filter for balancing. Sources
        that have no fragments remaining after the filter are included in the result and
        are recorded in the returned statistics. If a source contains sections
        smaller than `min_context_length`, all sections are retrieved without duplicates

        Args:
            sections: List of sections to select from
        Returns:
            tuple[list[ClinicalSection], _SelectionStats]: Selected sections and statistics
        """
        # Group sections by source
        sections_by_source: dict[str, list[ClinicalSection]] = defaultdict(list)
        for section in sections:
            source_key = section.filename.strip() if section.filename else section.document_title
            if not source_key:
                source_key = UNKNOWN_SOURCE_KEY
            sections_by_source[source_key].append(section)

        # Initialize statistics
        stats = _SelectionStats(sources_total=len(sections_by_source))
        selected: list[ClinicalSection] = []

        # Select sections
        for source_key, source_sections in sections_by_source.items():
            # Filter sections by minimum context length
            long_sections = [
                section
                for section in source_sections
                if len(section.content) >= self.min_context_length
            ]
            stats.sections_filtered_short += len(source_sections) - len(long_sections)

            # If no sections are left, skip the source
            if not long_sections:
                stats.sources_skipped_short += 1
                stats.skipped_sources.append(source_key)
                logger.warning(
                    f"Source '{source_key}': no sections with content length >= "
                    f"{self.min_context_length} (total sections: {len(source_sections)})"
                )
                continue

            # If number of sections less than requested number, use all sections without balancing
            if len(long_sections) < self.sections_per_pdf:
                source_selected = list(long_sections)
                self.rng.shuffle(source_selected)
                logger.info(
                    f"Source '{source_key}': only {len(long_sections)} long sections available "
                    f"(< {self.sections_per_pdf} requested), using all of them without balancing",
                )
                selected.extend(source_selected)
                continue

            # Group sections by section type
            sections_by_type: dict[str, list[ClinicalSection]] = defaultdict(list)
            for section in long_sections:
                sections_by_type[section.section_type].append(section)

            type_keys = list(sections_by_type.keys())

            # Distribute requested N as evenly as possible across section types
            base = self.sections_per_pdf // len(type_keys)
            remainder = self.sections_per_pdf % len(type_keys)
            counts_by_type = {section_type: base for section_type in type_keys}

            # Sample remainder section types
            for section_type in self.rng.sample(type_keys, k=remainder):
                counts_by_type[section_type] += 1

            # Select sections by section type
            source_selected: list[ClinicalSection] = []
            for section_type, target_count in counts_by_type.items():
                if target_count == 0:
                    continue
                pool = sections_by_type[section_type]
                effective_count = min(target_count, len(pool))
                if effective_count < target_count:
                    logger.info(
                        f"Source '{source_key}', section_type '{section_type}': pool has "
                        f"{len(pool)} sections (< quota {target_count}); taking all without reuse"
                    )
                source_selected.extend(self.rng.sample(pool, k=effective_count))

            # Shuffle sections
            self.rng.shuffle(source_selected)
            # Add sections to selected list
            selected.extend(source_selected)

        return selected, stats

    def _append_pairs(self, pairs: list[QAPair], output_file: Path) -> None:
        """
        Append QA pairs to JSONL file.

        Args:
            pairs: List of QA pairs to append to the file
            output_file: Path to the file to append the QA pairs to
        """
        with Path(output_file).open("a", encoding="utf-8") as f:
            f.writelines(pair.model_dump_json() + "\n" for pair in pairs)
