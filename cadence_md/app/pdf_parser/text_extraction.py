import importlib
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)

LAYOUT_HEADER_FONT_RATIO = 1.12
LAYOUT_HEADER_MAX_WORDS = 18


@dataclass(frozen=True)
class TextQuality:
    avg_chars_per_page: float
    empty_page_ratio: float
    score: float


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    total_pages: int
    mode_used: str
    quality: TextQuality
    layout_headers: list[str] = field(default_factory=list)


class HybridTextExtractor:
    """Single parser mode: native text with layout fallback."""

    def extract(self, pdf_path: Path) -> ExtractionResult:
        native_text, native_pages = self._extract_native(pdf_path)
        native_quality = self._assess_quality(native_text, native_pages)
        native_result = ExtractionResult(
            text=native_text,
            total_pages=native_pages,
            mode_used="native",
            quality=native_quality,
        )

        if not self._is_quality_poor(native_quality):
            return native_result

        layout_result = self._extract_layout(pdf_path)
        if layout_result and layout_result.quality.score > native_result.quality.score:
            return layout_result
        return native_result

    def _extract_native(self, pdf_path: Path) -> tuple[str, int]:
        reader = pypdf.PdfReader(pdf_path, strict=True)
        text_parts = []
        for page in reader.pages:
            text_parts.append((page.extract_text() or "").strip())
        return "\n".join(text_parts), len(reader.pages)

    def _extract_layout(self, pdf_path: Path) -> ExtractionResult | None:
        fitz = self._load_pymupdf()
        if fitz is None:
            return None

        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            logger.exception("Failed to open PDF via PyMuPDF: %s", pdf_path)
            return None

        lines: list[str] = []
        headers: list[str] = []
        total_pages = len(doc)
        for page in doc:
            page_lines, page_headers = self._extract_layout_page_lines(page)
            lines.extend(page_lines)
            headers.extend(page_headers)

        text = "\n".join(lines)
        quality = self._assess_quality(text, total_pages)
        doc.close()
        return ExtractionResult(
            text=text,
            total_pages=total_pages,
            mode_used="layout",
            quality=quality,
            layout_headers=headers,
        )

    def _extract_layout_page_lines(self, page) -> tuple[list[str], list[str]]:
        page_data = page.get_text("dict")
        line_records: list[tuple[float, float, str, float, bool]] = []
        font_sizes: list[float] = []
        for block in page_data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                y0 = min(span.get("bbox", [0, 0, 0, 0])[1] for span in spans)
                x0 = min(span.get("bbox", [0, 0, 0, 0])[0] for span in spans)
                size = max(float(span.get("size", 0.0)) for span in spans)
                is_bold = any("bold" in span.get("font", "").lower() for span in spans)
                line_records.append((y0, x0, text, size, is_bold))
                if size > 0:
                    font_sizes.append(size)

        line_records.sort(key=lambda item: (item[0], item[1]))
        lines = [item[2] for item in line_records]
        if not font_sizes:
            return lines, []

        body_font = statistics.median(font_sizes)
        headers: list[str] = []
        for _, _, text, size, is_bold in line_records:
            if not self._looks_like_numbered_header(text):
                continue
            font_ratio = size / body_font if body_font > 0 else 1.0
            short_line = len(text.split()) <= LAYOUT_HEADER_MAX_WORDS
            if font_ratio >= LAYOUT_HEADER_FONT_RATIO or (is_bold and short_line):
                headers.append(text)
        return lines, headers

    def _assess_quality(self, text: str, pages_count: int) -> TextQuality:
        pages = max(pages_count, 1)
        if not text.strip():
            return TextQuality(avg_chars_per_page=0.0, empty_page_ratio=1.0, score=0.0)

        page_parts = [part.strip() for part in text.split("\n")]
        non_empty_pages = [part for part in page_parts if part]
        empty_ratio = 1 - (len(non_empty_pages) / max(len(page_parts), 1))
        avg_chars = len(text) / pages
        score = avg_chars * (1 - empty_ratio)
        return TextQuality(avg_chars_per_page=avg_chars, empty_page_ratio=empty_ratio, score=score)

    def _is_quality_poor(self, quality: TextQuality) -> bool:
        return quality.avg_chars_per_page < 700 or quality.empty_page_ratio > 0.45

    def _looks_like_numbered_header(self, text: str) -> bool:
        return bool(re.match(r"^\s*\d{1,2}(?:\.\d{1,2}){0,4}\.?\s+\S+", text))

    def _load_pymupdf(self):
        try:
            return importlib.import_module("fitz")
        except Exception:
            logger.debug("PyMuPDF is not available")
            return None
