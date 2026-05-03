"""Central configuration constants for QA dataset generation."""

import os
from pathlib import Path

from dotenv import load_dotenv

from cadence_md.app.enums import QuestionType, SectionType

load_dotenv(dotenv_path=Path(".env.dev"), override=True)


# Context / sampling
DEFAULT_MAX_CONTEXT_LENGTH = 10000
DEFAULT_MIN_CONTEXT_LENGTH = 500
DEFAULT_SECTIONS_PER_PDF = 3

# LLM defaults
DEFAULT_MODEL_NAME = "GigaChat-2-Max"
DEFAULT_TEMPERATURE = 0.0

GIGACHAT_VERIFY_SSL_CERTS = False
GIGACHAT_API_KEY_ENV = "GIGACHAT_API_KEY"
GIGACHAT_API_KEY = os.getenv(GIGACHAT_API_KEY_ENV, "").strip()

# Retry (for requests to LLM)
MAX_LLM_RETRY_ATTEMPTS = 5
INITIAL_RETRY_DELAY_SEC = 1.0
MAX_RETRY_DELAY_SEC = 30.0
RETRY_JITTER_MAX_FACTOR = 0.5

# Validation (generated QA pair)
MIN_QUESTION_LENGTH = 10
MIN_ANSWER_LENGTH = 20
CONTEXT_FALLBACK_MAX_LEN = 1000

# JSONL output
INTERMEDIATE_SAVE_SECTION_INTERVAL = 5

# Progress bar
GENERATION_TQDM_DESC = "Generating QA"
GENERATION_TQDM_UNIT = "section"

# Section selection
UNKNOWN_SOURCE_KEY = "unknown_source"

# Mapping section types to question types
SECTION_QUESTION_TYPES = {
    SectionType.DEFINITION: [QuestionType.SIMPLE, QuestionType.REASONING],
    SectionType.SYMPTOMS: [QuestionType.SIMPLE, QuestionType.CONDITIONAL],
    SectionType.DIAGNOSIS: [QuestionType.SIMPLE, QuestionType.COMPARISON],
    SectionType.TREATMENT: [QuestionType.SIMPLE, QuestionType.CONDITIONAL, QuestionType.COMPARISON],
    SectionType.PREVENTION: [QuestionType.SIMPLE],
    SectionType.REHABILITATION: [QuestionType.SIMPLE],
}

# Report output (same directory as JSONL, `{stem}_generation_report.md`)
GENERATION_REPORT_FILENAME_SUFFIX = "_generation_report.md"


def generation_report_path(output_file: Path) -> Path:
    """Path to Markdown report next to `output_file`."""
    return output_file.with_name(f"{output_file.stem}{GENERATION_REPORT_FILENAME_SUFFIX}")
