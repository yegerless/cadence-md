import importlib
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pypdf

from cadence_md.app.pdf_parser.config import LAYOUT_HEADER_FONT_RATIO, LAYOUT_HEADER_MAX_WORDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextQuality:
    """Quality assessment of the extracted text"""

    avg_chars_per_page: float
    empty_page_ratio: float
    score: float


@dataclass(frozen=True)
class ExtractionResult:
    """Result of the text extraction"""

    text: str
    total_pages: int
    mode_used: str
    quality: TextQuality
    layout_headers: list[str] = field(default_factory=list)


class HybridTextExtractor:
    """
    Single parser mode: native text with layout fallback
    Uses native text extraction as primary method, with layout fallback for corrupted text layers.
    """

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """
        Extract text from PDF using hybrid approach
        First, try to extract text using native method.
        If the quality is poor, try to extract text using layout method.
        Return the result with the highest quality.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ExtractionResult: Result of the text extraction
        """
        native_result = self._extract_native(pdf_path)

        if not self._is_quality_poor(native_result.quality):
            return native_result

        layout_result = self._extract_layout(pdf_path)
        if layout_result and layout_result.quality.score > native_result.quality.score:
            return layout_result
        return native_result

    def _extract_native(self, pdf_path: Path) -> tuple[str, int]:
        """
        Extract text from PDF using native method
        Uses pypdf library to extract text from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            tuple[str, int]: Text and number of pages
        """
        reader = pypdf.PdfReader(pdf_path, strict=True)
        text_parts = []
        for page in reader.pages:
            text_parts.append((page.extract_text() or "").strip())
        native_text = "\n".join(text_parts)
        native_pages = len(reader.pages)

        native_quality = self._assess_quality(native_text, native_pages)

        return ExtractionResult(
            text=native_text,
            total_pages=native_pages,
            mode_used="native",
            quality=native_quality,
        )

    def _extract_layout(self, pdf_path: Path) -> ExtractionResult | None:
        """
        Extract text from PDF using layout method
        Uses PyMuPDF library to extract text from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ExtractionResult: Result of the text extraction
        """
        fitz = self._load_pymupdf()
        if fitz is None:
            return None

        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            logger.exception(f"Failed to open PDF via PyMuPDF: {pdf_path}")
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
        """
        Extract lines from a page using PyMuPDF

        Args:
            page: Page from pdf document to extract lines from

        Returns:
            tuple[list[str], list[str]]: Lines and headers
        """
        page_data = page.get_text("dict")
        line_records: list[tuple[float, float, str, float, bool]] = []
        font_sizes: list[float] = []
        for block in page_data.get("blocks", []):
            # skip non-text blocks
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                # skip empty lines
                if not spans:
                    continue
                text = "".join(span.get("text", "") for span in spans).strip()
                # skip empty text
                if not text:
                    continue
                # get bounding box of the line
                y0 = min(span.get("bbox", [0, 0, 0, 0])[1] for span in spans)
                x0 = min(span.get("bbox", [0, 0, 0, 0])[0] for span in spans)
                size = max(float(span.get("size", 0.0)) for span in spans)
                # check if the line is bold
                is_bold = any("bold" in span.get("font", "").lower() for span in spans)
                # add line to the list
                line_records.append((y0, x0, text, size, is_bold))
                # add font size to the list
                if size > 0:
                    font_sizes.append(size)

        # sort lines by y0 and x0
        line_records.sort(key=lambda item: (item[0], item[1]))
        # extract text from the lines
        lines = [item[2] for item in line_records]
        # if no font sizes, return lines and empty headers
        if not font_sizes:
            return lines, []

        # calculate the median font size
        body_font = statistics.median(font_sizes)
        # extract headers from the lines
        headers: list[str] = []
        for _, _, text, size, is_bold in line_records:
            # skip non-numbered headers
            if not self._looks_like_numbered_header(text):
                continue
            # calculate the font ratio
            font_ratio = size / body_font if body_font > 0 else 1.0
            # check if the line is short
            short_line = len(text.split()) <= LAYOUT_HEADER_MAX_WORDS
            # check if the line is a header
            if font_ratio >= LAYOUT_HEADER_FONT_RATIO or (is_bold and short_line):
                headers.append(text)
        return lines, headers

    def _assess_quality(self, text: str, pages_count: int) -> TextQuality:
        """
        Calculate the quality of the extracted text

        Args:
            text: Extracted text
            pages_count: Number of pages in the document

        Returns:
            TextQuality: Quality assessment of the extracted text
        """
        # calculate the number of pages
        pages = max(pages_count, 1)
        # if the text is empty, return a low quality
        if not text.strip():
            return TextQuality(avg_chars_per_page=0.0, empty_page_ratio=1.0, score=0.0)

        page_parts = [part.strip() for part in text.split("\n")]
        # calculate the number of non-empty pages
        non_empty_pages = [part for part in page_parts if part]
        empty_ratio = 1 - (len(non_empty_pages) / max(len(page_parts), 1))
        # calculate the average number of characters per page
        avg_chars = len(text) / pages
        # calculate the score
        score = avg_chars * (1 - empty_ratio)
        return TextQuality(avg_chars_per_page=avg_chars, empty_page_ratio=empty_ratio, score=score)

    def _is_quality_poor(self, quality: TextQuality) -> bool:
        """
        Check if the quality is poor

        Args:
            quality: Quality assessment of the extracted text

        Returns:
            bool: True if the quality is poor, False otherwise
        """
        return quality.avg_chars_per_page < 700 or quality.empty_page_ratio > 0.45

    def _looks_like_numbered_header(self, text: str) -> bool:
        """
        Check if the text looks like a numbered header

        Args:
            text: Text to check

        Returns:
            bool: True if the text looks like a numbered header, False otherwise
        """
        return bool(re.match(r"^\s*\d{1,2}(?:\.\d{1,2}){0,4}\.?\s+\S+", text))

    def _load_pymupdf(self):
        """
        Load PyMuPDF library
        Only used for layout extraction, optional dependency.

        Returns:
            module: PyMuPDF library
        """
        try:
            return importlib.import_module("fitz")
        except Exception:
            logger.debug("PyMuPDF is not available")
            return None
