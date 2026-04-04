import json
import logging
import os
from pathlib import Path
from random import choice
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat.chat_models import GigaChat
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cadence_md.app.enums import QuestionType, SectionType
from cadence_md.app.pdf_parser import ClinicalSection
from qa_dataset_generator.prompts import GENERATION_PROMPTS
from qa_dataset_generator.schemas import QAPair, QAResponsePair

# Init logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.dev", override=True)
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY", "")

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


class QAGeneratorNodes:
    """Nodes for LangGraph QA generation graph"""

    def __init__(self, llm, max_context_length: int = 5000):
        self.llm = llm
        self.max_context_length = max_context_length
        self.parser = JsonOutputParser(pydantic_object=QAResponsePair)

    def initialize_generation(self, state: QAGenerationState) -> QAGenerationState:
        """Node 1: Initialization - Defining Question Types"""
        section = state["section"]
        section_type = SectionType(section.section_type)

        question_types = SECTION_QUESTION_TYPES.get(section_type, [QuestionType.SIMPLE])

        state["question_type"] = choice([qt.value for qt in question_types])
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
            result = chain.invoke({"context": context})

            # Parse response
            if not isinstance(result, dict):
                raise ValueError(f"Unexpected result format: {type(result)}")

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

    def __init__(self, llm: GigaChat, max_context_length: int = 5000):
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

    def run(self, section: ClinicalSection) -> QAPair:
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
        max_context_length: int = 5000,
    ):
        # init LLM
        llm_kwargs = {
            "model": model_name,
            "verify_ssl_certs": False,
            "temperature": temperature,
        }

        if base_url:
            llm_kwargs["base_url"] = base_url

        if load_api_key:
            llm_kwargs["credentials"] = GIGACHAT_API_KEY

        self.llm = GigaChat(**llm_kwargs)

        # Build graph
        self.graph = QAGenerationGraph(llm=self.llm, max_context_length=max_context_length)

    def generate_from_sections_file(self, sections_file: Path, output_file: Path) -> list[QAPair]:
        """Generating QA from a file with sections"""
        logger.info(f"Loading sections from {sections_file}")

        # Load sections
        sections = []
        with Path(sections_file).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sections.append(ClinicalSection.from_dict(data))

        # TODO: random choice sections subset

        logger.info(f"Loaded {len(sections)} sections")

        # Generate QA for each section
        all_pairs = []

        for i, section in enumerate(sections, 1):
            logger.info(f"Section processing {i}/{len(sections)}")

            try:
                pair = self.graph.run(section)
                all_pairs.append(pair)

                # Intermediate save every 10 sections
                if i % 10 == 0:
                    self._save_pairs(all_pairs, output_file)
                    logger.info(f"Intermediate storage: {len(all_pairs)} pairs")

            except Exception as e:
                logger.error(f"Error processing section {section.section_title}: {e}")
                continue

        # Final save
        self._save_pairs(all_pairs, output_file)
        logger.info(f"Generation complete: {len(all_pairs)} QA pairs")

        return all_pairs

    def _save_pairs(self, pairs: list[QAPair], output_file: Path):
        """Saving QA pairs in JSON"""
        with Path(output_file).open("w", encoding="utf-8") as f:
            f.writelines(pair.model_dump_json() + "\n" for pair in pairs)
