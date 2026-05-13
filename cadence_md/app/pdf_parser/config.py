"""Tunable parameters for the clinical PDF parser and hybrid text extraction."""

MAX_SECTION_LENGTH = 10_000
MIN_SECTION_LENGTH = 100
TOC_WINDOW_CHARS = 60_000
TOC_ENTRY_MAX_GAP = 2_500
TOC_MIN_CLUSTER_SIZE = 8
TOC_SANITY_WINDOW = 30_000
MIN_TRIM_START = 6_000
CYRILLIC_RATIO_FLOOR = 0.15
MIN_LETTERS_FOR_ENCODING_CHECK = 300

HEADER_OCR_NORMALIZATION = {
    "реабилитау": "реабилитац",
    "орагнизац": "организац",
    "лечени ": "лечение ",
    "диагности ": "диагностика ",
}

LAYOUT_HEADER_FONT_RATIO = 1.12
LAYOUT_HEADER_MAX_WORDS = 18
