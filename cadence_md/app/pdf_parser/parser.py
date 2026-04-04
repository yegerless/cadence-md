import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pypdf
from tqdm import tqdm

from cadence_md.app.pdf_parser.regexps import (
    EXCLUDE_PATTERNS,
    ICD_RE,
    SECTION_HEADER_RE,
    SECTION_PATTERNS,
    STOP_RE,
)

logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)


# TODO: refactor to pydantic
@dataclass
class ClinicalSection:
    filename: str
    document_title: str
    section_type: str
    section_title: str
    content: str
    mkb_codes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class ClinicalGuidelinesParser:
    """
    Custom PDF parser for clinical recommendation texts.
    """

    def __init__(self):
        self.section_patterns = SECTION_PATTERNS
        self.exclude_patterns = EXCLUDE_PATTERNS

    def parse_directory(self, path: Path) -> list[ClinicalSection]:
        """
        Method for parsing all pdf files in directory
        """
        pdf_files = list(Path(path).glob("*.pdf"))

        result = []
        with tqdm(pdf_files) as pbar:
            for pdf_file in pbar:
                pbar.set_description(f"{pdf_file.name}")
                parsed_pdf = self.parse_pdf(pdf_file)
                if parsed_pdf:
                    result.extend(parsed_pdf)
        return result

    def parse_pdf(self, pdf_path: Path) -> list[ClinicalSection] | None:
        """
        Method for parsing pdf files.
        """
        try:
            text, _ = self._extract_text_with_page_markers(pdf_path)

            title = self._extract_document_title(text, pdf_path)
            mkb_codes = self._extract_mkb_codes(text)

            raw_sections = self._extract_sections(text)
        except KeyError as e:
            print(f"KeyError exception with file {pdf_path}: {e}")
            return None

        result = []
        for sec_title, sec_body in raw_sections:
            sec_type = self._classify_section(sec_title)
            if not sec_type:
                continue
            cleaned = self._clean_section_text(sec_body)
            if len(cleaned) < 100:  # discard very short
                continue
            section = ClinicalSection(
                filename=pdf_path.name,
                document_title=title,
                section_type=sec_type,
                section_title=sec_title,
                content=cleaned[:1000000],  # max text fragment length
                mkb_codes=mkb_codes,
            )
            result.append(section)

        return result

    def _extract_text_with_page_markers(self, pdf_path: Path) -> tuple[str, int]:
        reader = pypdf.PdfReader(pdf_path, strict=True)
        text_parts = []
        for i, page in enumerate(reader.pages):
            if i < 3:
                continue
            page_text = page.extract_text() or ""
            if i > 10 and STOP_RE.search(page_text):
                break
            text_parts.append(f"{page_text}\n")
        full_text = "".join(text_parts)
        return full_text, len(reader.pages)

    def _extract_document_title(self, text: str, pdf_path: Path) -> str:
        header = text[:1500]
        # try get title after phrase "клинические рекомендации"
        m = re.search(
            r"клинические\s+рекомендации[:\s]*\n*([А-ЯЁа-яё][А-ЯЁа-яё\s\-,()]+?)(?:\n|МКБ)",
            header,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
            return title[:500]
        # fallback: file name
        file_name = Path(pdf_path).name
        return " ".join(file_name.split("_"))

    def _extract_mkb_codes(self, text: str) -> list[str]:
        header = text[:3000]
        codes = ICD_RE.findall(header)
        # delete duplicates
        seen = set()
        out = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _extract_sections(self, text: str) -> list[tuple[str, str]]:
        sections = []
        matches = list(SECTION_HEADER_RE.finditer(text))
        for i, m in enumerate(matches):
            number = m.group(1).strip()
            title = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i < len(matches) - 1 else len(text)
            body = text[start:end]
            sections.append((f"{number} {title}", body))
        return sections

    def _classify_section(self, section_title: str) -> str | None:
        title_lower = section_title.lower()
        # exclude
        for p in self.exclude_patterns:
            if re.search(p, title_lower):
                return None
        # include
        for sec_type, patterns in self.section_patterns.items():
            for p in patterns:
                if re.search(p, title_lower):
                    return sec_type
        return None

    def _clean_section_text(self, text: str) -> str:
        # delete long sequence of points
        text = re.sub(r"\.{3,}", " ", text)
        # delete strings like «... 23» from content
        text = re.sub(r"\n.*\s\d+\s*$", "", text, flags=re.MULTILINE)
        # delete links to tables/figures
        text = re.sub(r"[Тт]аблиц[ау]\s+\d+[^\n.]*\.?", " ", text)
        text = re.sub(r"[Рр]исуно[кч]\s+\d+[^\n.]*\.?", " ", text)
        text = re.sub(r"см\.\s*табл[^\n]*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"см\.\s*рис[^\n]*", " ", text, flags=re.IGNORECASE)
        # delete links like [1], [2,3]
        text = re.sub(r"\[\d+(?:[,;\s]*\d+)*\]", " ", text)
        # normalize spaces
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()
