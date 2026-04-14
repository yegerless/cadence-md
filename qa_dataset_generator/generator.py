import json
import logging
import os
import random
import time
from collections import defaultdict
from http import HTTPStatus
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from gigachat.exceptions import ResponseError
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat.chat_models import GigaChat
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cadence_md.app.enums import QuestionType, SectionType
from cadence_md.app.pdf_parser import ClinicalSection
from qa_dataset_generator.prompts import GENERATION_PROMPTS
from qa_dataset_generator.schemas import QAPair, QAResponsePair

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_LENGTH = 10_000
DEFAULT_SECTIONS_PER_PDF = 3
MAX_LLM_RETRY_ATTEMPTS = 6
INITIAL_RETRY_DELAY_SEC = 1.0
MAX_RETRY_DELAY_SEC = 30.0

# Mapping section types to question types
SECTION_QUESTION_TYPES = {
    SectionType.DEFINITION: [QuestionType.SIMPLE, QuestionType.REASONING],
    SectionType.SYMPTOMS: [QuestionType.SIMPLE, QuestionType.CONDITIONAL],
    SectionType.DIAGNOSIS: [QuestionType.SIMPLE, QuestionType.COMPARISON],
    SectionType.TREATMENT: [QuestionType.SIMPLE, QuestionType.CONDITIONAL, QuestionType.COMPARISON],
    SectionType.PREVENTION: [QuestionType.SIMPLE],
    SectionType.REHABILITATION: [QuestionType.SIMPLE],
}


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
                "Temporary LLM error, retrying in %.1f sec (%s/%s): %s",
                delay,
                attempt,
                max_attempts,
                error,
            )
            jitter = random.uniform(0, 0.5 * delay)
            time.sleep(delay + jitter)
            delay = min(MAX_RETRY_DELAY_SEC, delay * 2)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry loop finished without result or error")


class QAGeneratorNodes:
    """Nodes for LangGraph QA generation graph"""

    def __init__(self, llm: GigaChat, max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH):
        self.llm = llm
        self.max_context_length = max_context_length
        self.parser = JsonOutputParser(pydantic_object=QAResponsePair)

    def initialize_generation(self, state: QAGenerationState) -> QAGenerationState:
        """Node 1: Initialization - Defining Question Types"""
        section = state["section"]
        section_type = SectionType(section.section_type)

        question_types = SECTION_QUESTION_TYPES.get(section_type, [QuestionType.SIMPLE])

        state["question_type"] = random.choice([qt.value for qt in question_types])
        state["generated_pair"] = None
        state["errors"] = []

        logger.info(
            f"Initialization for '{section.section_type}', "
            f"types of questions: {state['question_type']}"
        )

        return state

    def generate_questions(self, state: QAGenerationState) -> QAGenerationState:
        """Node 2: Question generation via LLM"""
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

            # Convert QAPair
            pair = QAPair(
                question=result.get("question", ""),
                answer=result.get("answer", ""),
                question_type=question_type,
                section_type=section.section_type,
                document_title=section.document_title,
                section_title=section.section_title,
                mkb_codes=section.mkb_codes,
                context=result.get("context", context[:1000]),
            )
            state["generated_pair"] = pair

            logger.info("Generated QA pair")

        except Exception as e:
            error_msg = f"Generation error: {e}"
            logger.error(error_msg)
            state["errors"].append(error_msg)

        return state

    def validate_qa_pair(self, state: QAGenerationState) -> QAGenerationState:
        """Node 3: Validation of QA pairs"""
        invalid = False
        pair = state["generated_pair"]

        # Base checks
        if not pair:
            state["errors"].append("generated_pair not found")
            invalid = True
        elif len(pair.question) < 10:
            state["errors"].append(f"Question too short: {pair.question}")
            state["generated_pair"] = None
        elif len(pair.answer) < 20:
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
    """LangGraph graph for generating QA datasets"""

    def __init__(self, llm: GigaChat, max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH):
        self.nodes = QAGeneratorNodes(llm, max_context_length)
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Construction of a graph"""
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
        """Running a graph for one section"""
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
    """Synthetic dataset generator for all sections"""

    def __init__(
        self,
        model_name: str = "GigaChat",  # "GigaChat-2-Max"
        base_url: str | None = None,  # for local models
        load_api_key: bool = True,
        temperature: float = 0.7,
        max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
        sections_per_pdf: int = DEFAULT_SECTIONS_PER_PDF,
        seed: int | None = None,
    ):
        if sections_per_pdf <= 0:
            raise ValueError("sections_per_pdf must be greater than 0")
        self.sections_per_pdf = sections_per_pdf
        self.rng = random.Random(seed)

        load_dotenv(dotenv_path=".env.dev", override=True)

        # init LLM
        llm_kwargs = {
            "model": model_name,
            "verify_ssl_certs": False,
            "temperature": temperature,
        }

        if base_url:
            llm_kwargs["base_url"] = base_url

        if load_api_key:
            llm_kwargs["credentials"] = self._get_api_key()

        self.llm = GigaChat(**llm_kwargs)

        # Build graph
        self.graph = QAGenerationGraph(llm=self.llm, max_context_length=max_context_length)

    def _get_api_key(self) -> str:
        """Get API key after env loading."""
        api_key = os.getenv("GIGACHAT_API_KEY", "")
        if not api_key:
            raise ValueError("GIGACHAT_API_KEY is not set while load_api_key=True")
        return api_key

    def generate_from_sections_file(self, sections_file: Path, output_file: Path) -> list[QAPair]:
        """Generating QA from a file with sections"""
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
                        logger.warning(
                            "Skipping invalid JSON at line %s: %s",
                            line_number,
                            error,
                        )
                        continue
                    try:
                        sections.append(ClinicalSection.from_dict(data))
                    except (TypeError, ValueError) as error:
                        skipped_invalid_jsonl += 1
                        logger.warning(
                            "Skipping invalid section at line %s: %s",
                            line_number,
                            error,
                        )
                        continue
        selected_sections = self._select_random_sections(sections)
        logger.info(
            "Loaded %s sections, selected %s for generation, skipped_invalid_jsonl=%s",
            len(sections),
            len(selected_sections),
            skipped_invalid_jsonl,
        )

        # Generate QA for selected sections
        all_pairs: list[QAPair] = []
        pending_pairs: list[QAPair] = []
        processed = 0
        failed = 0

        for i, section in enumerate(selected_sections, 1):
            logger.info(f"Section processing {i}/{len(selected_sections)}")
            processed += 1

            try:
                pair = self.graph.run(section)
                if pair is None:
                    failed += 1
                    continue
                all_pairs.append(pair)
                pending_pairs.append(pair)

                # Intermediate save every 10 sections
                if i % 10 == 0 and pending_pairs:
                    self._append_pairs(pending_pairs, output_file)
                    logger.info(
                        "Intermediate storage: %s total generated, %s failed",
                        len(all_pairs),
                        failed,
                    )
                    pending_pairs = []

            except Exception as e:
                failed += 1
                logger.error(f"Error processing section {section.section_title}: {e}")
                continue

        # Final save
        if pending_pairs:
            self._append_pairs(pending_pairs, output_file)
        logger.info(
            "Generation complete: processed=%s generated=%s failed=%s skipped_invalid_jsonl=%s",
            processed,
            len(all_pairs),
            failed,
            skipped_invalid_jsonl,
        )

        return all_pairs

    def _select_random_sections(self, sections: list[ClinicalSection]) -> list[ClinicalSection]:
        """Select N sections per source with balanced section_type distribution."""
        sections_by_source: dict[str, list[ClinicalSection]] = defaultdict(list)
        for section in sections:
            source_key = section.filename.strip() if section.filename else section.document_title
            if not source_key:
                source_key = "unknown_source"
            sections_by_source[source_key].append(section)

        selected: list[ClinicalSection] = []
        for source_key, source_sections in sections_by_source.items():
            sections_by_type: dict[str, list[ClinicalSection]] = defaultdict(list)
            for section in source_sections:
                sections_by_type[section.section_type].append(section)

            type_keys = list(sections_by_type.keys())
            if not type_keys:
                continue

            # Distribute requested N as evenly as possible across section types.
            base = self.sections_per_pdf // len(type_keys)
            remainder = self.sections_per_pdf % len(type_keys)
            counts_by_type = {section_type: base for section_type in type_keys}

            for section_type in self.rng.sample(type_keys, k=remainder):
                counts_by_type[section_type] += 1

            source_selected: list[ClinicalSection] = []
            for section_type, target_count in counts_by_type.items():
                if target_count == 0:
                    continue
                pool = list(sections_by_type[section_type])
                if target_count <= len(pool):
                    source_selected.extend(self.rng.sample(pool, k=target_count))
                    continue

                # If there are fewer sections than required quota,
                # reuse random sections of that type.
                source_selected.extend(pool)
                missing = target_count - len(pool)
                source_selected.extend(self.rng.choices(pool, k=missing))
                logger.info(
                    "Reused %s sections for source '%s' and section_type '%s' "
                    "to satisfy balanced quota (%s requested, %s available).",
                    missing,
                    source_key,
                    section_type,
                    target_count,
                    len(pool),
                )

            self.rng.shuffle(source_selected)
            selected.extend(source_selected)
        return selected

    def _append_pairs(self, pairs: list[QAPair], output_file: Path) -> None:
        """Append QA pairs to JSONL file."""
        with Path(output_file).open("a", encoding="utf-8") as f:
            f.writelines(pair.model_dump_json() + "\n" for pair in pairs)
