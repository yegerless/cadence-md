import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from cadence_md.app.pdf_parser.regexps import (
    EXCLUDE_PATTERNS,
    HEADER_CANDIDATE_RE,
    HEADER_KEYWORDS,
    ICD_RE,
    SECTION_PATTERNS,
    STOP_RE,
    TOC_ENTRY_RE,
)
from cadence_md.app.pdf_parser.text_extraction import HybridTextExtractor

logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class HeaderMatch:
    number: str
    title: str
    start: int
    end: int
    level: int


class ClinicalGuidelinesParser:
    """
    Custom PDF parser for clinical recommendation texts.
    """

    def __init__(
        self,
        extractor: HybridTextExtractor | None = None,
    ):
        self.section_patterns = SECTION_PATTERNS
        self.exclude_patterns = EXCLUDE_PATTERNS
        self.extractor = extractor or HybridTextExtractor()

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
            text, _, layout_headers = self._extract_text_with_page_markers(pdf_path)
            text = self._normalize_extracted_text(text)

            title = self._extract_document_title(text, pdf_path)
            mkb_codes = self._extract_mkb_codes(text)

            raw_sections = self._extract_sections(text, layout_headers=layout_headers)
        except KeyError:
            logger.exception("KeyError while parsing file %s", pdf_path)
            return None

        result = []
        root_section_types: dict[str, str] = {}
        for number, title_part, sec_body in raw_sections:
            sec_title = f"{number} {title_part}"
            if self._should_skip_section_title(sec_title):
                continue
            sec_type = self._classify_section(sec_title)
            root_number = number.split(".")[0]
            if number.count(".") == 0 and sec_type:
                root_section_types[root_number] = sec_type
            if not sec_type and root_number != "1":
                sec_type = root_section_types.get(root_number)
            if not sec_type:
                continue
            cleaned = self._clean_section_text(sec_body)
            if len(cleaned) < MIN_SECTION_LENGTH:
                continue
            chunks = self._split_large_section(cleaned, number)
            for idx, chunk in enumerate(chunks, start=1):
                section_title = sec_title
                if len(chunks) > 1:
                    section_title = f"{sec_title} [part {idx}/{len(chunks)}]"
                section = ClinicalSection(
                    filename=pdf_path.name,
                    document_title=title,
                    section_type=sec_type,
                    section_title=section_title,
                    content=chunk,
                    mkb_codes=mkb_codes,
                )
                result.append(section)

        return result

    def _should_skip_section_title(self, section_title: str) -> bool:
        """Return True for section titles that must be excluded from output."""
        normalized = self._normalize_header_text(section_title.lower())
        compact = re.sub(r"[\s\W_]+", "", normalized)

        if re.search(r"\bэпидемиолог", normalized):
            return True
        if re.search(r"стратификац\w*\s+риска\s+хирургическ\w*\s+лечени\w*", normalized):
            return True

        # Robust matching for OCR/PDF extraction glitches with missing spaces:
        # "илисостояния", "поМеждународной", etc.
        coding_compact_anchor = "особенностикодированиязаболеванияилисостояния"
        return coding_compact_anchor in compact

    def _extract_text_with_page_markers(self, pdf_path: Path) -> tuple[str, int, list[str]]:
        extraction = self.extractor.extract(pdf_path)
        text = extraction.text
        total_pages = extraction.total_pages
        mode_used = extraction.mode_used
        quality_score = extraction.quality.score
        layout_headers = extraction.layout_headers

        if extraction.mode_used == "layout" and self._is_text_corrupted(extraction.text):
            native_variant = self._try_extract_native_variant(pdf_path)
            if native_variant:
                native_text, native_pages, native_score = native_variant
                if self._is_better_text_variant(
                    current_text=extraction.text,
                    current_score=quality_score,
                    candidate_text=native_text,
                    candidate_score=native_score,
                ):
                    text = native_text
                    total_pages = native_pages
                    mode_used = "native_fallback"
                    quality_score = native_score
                    layout_headers = []
                    logger.warning(
                        "Detected corrupted layout text layer for %s; fallback to native extraction",  # noqa: E501
                        pdf_path.name,
                    )

        if self._is_text_corrupted(text):
            logger.warning("Detected corrupted text layer for %s", pdf_path.name)

        logger.info(
            "Extracted %s with mode=%s pages=%s quality=%.2f",
            pdf_path.name,
            mode_used,
            total_pages,
            quality_score,
        )
        return text, total_pages, layout_headers

    def _try_extract_native_variant(self, pdf_path: Path) -> tuple[str, int, float] | None:
        if not hasattr(self.extractor, "_extract_native"):
            return None
        try:
            native_text, native_pages = self.extractor._extract_native(pdf_path)
        except Exception:
            logger.exception("Native fallback extraction failed for %s", pdf_path)
            return None

        if hasattr(self.extractor, "_assess_quality"):
            quality = self.extractor._assess_quality(native_text, native_pages)
            return native_text, native_pages, quality.score
        pages = max(native_pages, 1)
        return native_text, native_pages, len(native_text) / pages

    def _is_better_text_variant(
        self,
        current_text: str,
        current_score: float,
        candidate_text: str,
        candidate_score: float,
    ) -> bool:
        current_corrupted = self._is_text_corrupted(current_text)
        candidate_corrupted = self._is_text_corrupted(candidate_text)
        if current_corrupted != candidate_corrupted:
            return not candidate_corrupted

        current_ratio = self._cyrillic_ratio(current_text)
        candidate_ratio = self._cyrillic_ratio(candidate_text)
        if candidate_ratio > current_ratio + 0.05:
            return True
        if current_ratio > candidate_ratio + 0.05:
            return False
        if candidate_score > current_score + 1:
            return True
        return len(candidate_text) > len(current_text)

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

    def _extract_sections(
        self, text: str, layout_headers: list[str] | None = None
    ) -> list[tuple[str, str, str]]:
        sections: list[tuple[str, str, str]] = []
        toc_end = self._detect_toc_end(text)
        matches = self._extract_header_matches(text)
        if layout_headers:
            matches = self._merge_layout_headers(matches, text, layout_headers)
        matches = [m for m in matches if m.start >= toc_end]

        if not matches:
            return sections

        trim_end = self._find_trim_end(text, matches)
        matches = [m for m in matches if m.start < trim_end]

        for i, header in enumerate(matches):
            start = header.end
            end = matches[i + 1].start if i < len(matches) - 1 else trim_end
            body = text[start:end].strip()
            if body:
                sections.append((header.number, header.title, body))
        return sections

    def _merge_layout_headers(
        self, regex_headers: list[HeaderMatch], text: str, layout_headers: list[str]
    ) -> list[HeaderMatch]:
        merged = list(regex_headers)
        occupied_ranges = [(header.start, header.end) for header in regex_headers]
        for header_line in layout_headers:
            normalized_line = self._normalize_header_text(header_line)
            match = re.match(
                r"^\s*(?P<number>\d{1,2}(?:\.\d{1,2}){0,4}\.?)\s+(?P<title>.+)$",
                normalized_line,
            )
            if not match:
                continue
            number = match.group("number").rstrip(".")
            title = self._normalize_header_text(match.group("title"))
            if not self._is_likely_header(number, title):
                continue
            candidates = [normalized_line, f"{number} {title}"]
            start = -1
            matched_candidate = ""
            for candidate in candidates:
                pos = text.find(candidate)
                if pos != -1:
                    start = pos
                    matched_candidate = candidate
                    break
            if start == -1:
                continue
            end = start + len(matched_candidate)
            if self._range_overlaps(start, end, occupied_ranges):
                continue
            occupied_ranges.append((start, end))
            merged.append(
                HeaderMatch(
                    number=number,
                    title=title,
                    start=start,
                    end=end,
                    level=self._header_level(number),
                )
            )
        merged.sort(key=lambda item: item.start)
        return merged

    def _range_overlaps(self, start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
        return any(start <= right and end >= left for left, right in ranges)

    def _classify_section(self, section_title: str) -> str | None:
        title_lower = self._normalize_header_text(section_title.lower())
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
        # delete TOC-like strings with trailing page numbers
        text = re.sub(
            r"(?m)^\s*\d{1,2}(?:\.\d{1,2}){0,4}\.?\s+[^\n]{3,180}\s(?:\.{2,}\s*|\s+)\d{1,3}\s*$",
            "",
            text,
        )
        # delete links to tables/figures
        text = re.sub(r"[Тт]аблиц[ау]\s+\d+[^\n.]*\.?", " ", text)
        text = re.sub(r"[Рр]исуно[кч]\s+\d+[^\n.]*\.?", " ", text)
        text = re.sub(r"см\.\s*табл[^\n]*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"см\.\s*рис[^\n]*", " ", text, flags=re.IGNORECASE)
        # delete links like [1], [2,3]
        text = re.sub(r"\[\d+(?:[,;\s]*\d+)*\]", " ", text)
        # normalize spaces
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_extracted_text(self, text: str) -> str:
        text = text.replace("\r", "\n").replace("\xa0", " ")
        text = re.sub(r"([А-Яа-яA-Za-z])-\n([А-Яа-яA-Za-z])", r"\1\2", text)
        text = re.sub(r"(?m)^(\d{1,2})\s*[,;]\s*([А-ЯЁA-Zа-яёa-z])", r"\1. \2", text)
        text = re.sub(
            r"(?m)^(\d{1,2}(?:\.\d{1,2}){0,4}\.?)\s*\n\s*([А-ЯЁA-Z])",
            r"\1 \2",
            text,
        )
        text = re.sub(
            r"([^\n])\s+(?=(?:[1-7](?:\.\d{1,2}){0,4}\.?)\s+[А-ЯЁA-Z][^\n]{3,220}(?:\s+\d{1,2}(?:\.\d{1,2}){0,4}\.?\s+[А-ЯЁA-Z]|$))",
            r"\1\n",
            text,
        )
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text)

    def _normalize_header_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        normalized = re.sub(r"\s(?:\.{2,}\s*|\s+)\d{1,3}\s*$", "", normalized)
        for source, target in HEADER_OCR_NORMALIZATION.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _header_level(self, number: str) -> int:
        return number.count(".") + 1

    def _top_level_number(self, number: str) -> int | None:
        head = number.split(".")[0]
        if not head.isdigit():
            return None
        return int(head)

    def _extract_header_matches(self, text: str) -> list[HeaderMatch]:
        headers: list[HeaderMatch] = []
        seen_keys: set[tuple[str, int]] = set()

        def add_header(candidate: HeaderMatch) -> None:
            dedup_key = (candidate.number, candidate.start // 100)
            if dedup_key in seen_keys:
                return
            seen_keys.add(dedup_key)
            headers.append(candidate)

        for match in HEADER_CANDIDATE_RE.finditer(text):
            number = match.group("number").rstrip(".")
            title = self._normalize_header_text(match.group("title"))
            nested = re.match(r"^(?P<number>\d{1,2}(?:\.\d{1,2}){0,4}\.?)\s+(?P<title>.+)$", title)
            if nested:
                number = nested.group("number").rstrip(".")
                title = self._normalize_header_text(nested.group("title"))
            if not self._is_likely_header(number, title):
                continue
            add_header(
                HeaderMatch(
                    number=number,
                    title=title,
                    start=match.start(),
                    end=match.end(),
                    level=self._header_level(number),
                )
            )
        for inline_header in self._extract_inline_header_matches(text):
            add_header(inline_header)
        headers.extend(self._extract_unnumbered_major_headers(text, headers))
        headers.sort(key=lambda x: x.start)
        return headers

    def _extract_inline_header_matches(self, text: str) -> list[HeaderMatch]:
        headers: list[HeaderMatch] = []
        split_re = re.compile(r"\d{1,2}(?:\.\d{1,2}){0,4}\.?\s+[А-ЯЁA-Z]")
        inline_re = re.compile(
            r"(?P<number>\d{1,2}(?:\.\d{1,2}){0,4}\.?)\s+"
            r"(?P<title>[^\n]{4,260}?)(?=(?:\s+\d{1,2}(?:\.\d{1,2}){0,4}\.?\s+[А-ЯЁA-Z])|$)"
        )

        for line_match in re.finditer(r"[^\n]+", text):
            line = line_match.group(0)
            if len(split_re.findall(line)) < 2:
                continue
            for candidate in inline_re.finditer(line):
                number = candidate.group("number").rstrip(".")
                title = self._normalize_header_text(candidate.group("title"))
                if not self._is_likely_header(number, title):
                    continue
                start = line_match.start() + candidate.start()
                headers.append(
                    HeaderMatch(
                        number=number,
                        title=title,
                        start=start,
                        end=line_match.start() + candidate.end(),
                        level=self._header_level(number),
                    )
                )
        return headers

    def _extract_unnumbered_major_headers(
        self, text: str, existing_headers: list[HeaderMatch]
    ) -> list[HeaderMatch]:
        extra_headers: list[HeaderMatch] = []
        patterns = [
            (
                "4",
                re.compile(
                    r"(?mi)^\s*(медицинск\w*\s+реабилитац\w*[^\n]{10,220})\s*$",
                ),
            ),
            (
                "5",
                re.compile(
                    r"(?mi)^\s*([^\n]{0,40}диспансерн\w*[^\n]{10,220})\s*$",
                ),
            ),
        ]

        for number, pattern in patterns:
            for match in pattern.finditer(text):
                title = self._normalize_header_text(match.group(1))
                if not self._is_likely_header(number, title):
                    continue
                if self._has_nearby_same_top_header(
                    number,
                    match.start(),
                    existing_headers,
                    extra_headers,
                ):
                    continue
                extra_headers.append(
                    HeaderMatch(
                        number=number,
                        title=title,
                        start=match.start(),
                        end=match.end(),
                        level=self._header_level(number),
                    )
                )
        return extra_headers

    def _has_nearby_same_top_header(
        self,
        number: str,
        position: int,
        existing_headers: list[HeaderMatch],
        extra_headers: list[HeaderMatch],
    ) -> bool:
        for header in [*existing_headers, *extra_headers]:
            if not header.number.startswith(number):
                continue
            if abs(header.start - position) < 400:
                return True
        return False

    def _is_likely_header(self, number: str, title: str) -> bool:
        clean_title = self._normalize_header_text(title) if title else ""
        clean_title_lower = clean_title.lower()
        is_valid_length = 4 <= len(clean_title_lower) <= 260
        is_list_item = bool(re.match(r"^\d+[).]\s+", clean_title_lower))
        starts_with_punct_noise = bool(re.match(r"^[^\wа-яёa-z]*[;:.,”\"'`]+", clean_title_lower))
        basic_valid = bool(number and title) and is_valid_length and not is_list_item
        if not basic_valid or starts_with_punct_noise:
            return False

        if re.search(r"\s{2,}", clean_title_lower):
            clean_title_lower = re.sub(r"\s+", " ", clean_title_lower)

        if re.match(r"^(рисунок|таблица|комментар\b|коммент\s*ар)", clean_title_lower):
            return False

        top_level_number = self._top_level_number(number)
        if top_level_number is None or top_level_number > 7:
            return False

        if "." in number:
            starts_like_title = bool(re.match(r"^[\"«(]?[А-ЯЁA-Zа-яёa-z]", clean_title))
        else:
            starts_like_title = bool(re.match(r"^[\"«(]?[А-ЯЁA-Z]", clean_title))
        has_known_keyword = any(
            re.search(keyword, clean_title_lower) for keyword in HEADER_KEYWORDS
        )
        matches_stop = bool(re.search(STOP_RE, clean_title_lower))
        if matches_stop or has_known_keyword:
            top_level_ok = "." in number or self._is_plausible_top_level_title(clean_title_lower)
            return starts_like_title and top_level_ok

        word_count = len(clean_title_lower.split())
        if "." in number:
            has_forbidden_prefix = bool(
                re.match(r"^(рисунок|таблица|комментарий)\b", clean_title_lower)
            )
            return word_count <= 16 and not has_forbidden_prefix

        return False

    def _is_plausible_top_level_title(self, title: str) -> bool:
        starts_with_anchor = bool(
            re.match(
                (
                    r"^(кратк|диагностик|лечени|терапи|"
                    r"медицинск\w*\s+реабилитац|профилактик|диспансер|"
                    r"организац|дополнительн)"
                ),
                title,
            )
        )
        contains_dispansary_phrase = bool(re.search(r"диспансер\w*\s+наблюден\w*", title))
        return starts_with_anchor or contains_dispansary_phrase

    def _detect_toc_end(self, text: str) -> int:
        toc_window = text[:TOC_WINDOW_CHARS]
        toc_entries = list(TOC_ENTRY_RE.finditer(toc_window))
        if len(toc_entries) >= 8:
            cluster_end = self._toc_end_from_entries(toc_entries)
            if cluster_end and self._has_body_anchors_after_toc(text, cluster_end):
                return cluster_end

        # Fallback for PDFs where TOC lines do not contain page-number tails.
        # We detect the second top-level "1 ..." occurrence.
        # In guideline PDFs this usually marks transition from TOC to real body.
        early_headers = [
            h for h in self._extract_header_matches(toc_window) if h.start < TOC_WINDOW_CHARS
        ]
        if len(early_headers) < 12:
            return 0

        top_level_one = [h for h in early_headers if h.number == "1"]
        if len(top_level_one) >= 2:
            first, second = top_level_one[0], top_level_one[1]
            if second.start - first.start > 300:
                fallback_end = second.start
                if self._has_body_anchors_after_toc(text, fallback_end):
                    return fallback_end
        return 0

    def _toc_end_from_entries(self, toc_entries: list[re.Match[str]]) -> int:
        cluster_count = 1
        cluster_end = toc_entries[0].end()
        best_cluster_count = 1
        best_cluster_end = cluster_end

        for idx in range(1, len(toc_entries)):
            gap = toc_entries[idx].start() - toc_entries[idx - 1].start()
            if gap <= TOC_ENTRY_MAX_GAP:
                cluster_count += 1
                cluster_end = toc_entries[idx].end()
            else:
                if cluster_count > best_cluster_count:
                    best_cluster_count = cluster_count
                    best_cluster_end = cluster_end
                cluster_count = 1
                cluster_end = toc_entries[idx].end()

        if cluster_count > best_cluster_count:
            best_cluster_count = cluster_count
            best_cluster_end = cluster_end

        if best_cluster_count >= TOC_MIN_CLUSTER_SIZE:
            return best_cluster_end
        return 0

    def _has_body_anchors_after_toc(self, text: str, toc_end: int) -> bool:
        window_end = min(len(text), toc_end + TOC_SANITY_WINDOW)
        anchors_window = text[toc_end:window_end]
        anchors = self._extract_header_matches(anchors_window)
        core_headers = 0
        for header in anchors:
            top_level = self._top_level_number(header.number)
            if top_level in {1, 2, 3, 4, 5}:
                core_headers += 1
        return core_headers >= 5

    def _is_prevention_header(self, title: str) -> bool:
        return bool(re.search(r"профилактик|диспансер|наблюдени", title, flags=re.IGNORECASE))

    def _is_stop_header(self, title: str) -> bool:
        if STOP_RE.search(title):
            return True
        return bool(
            re.search(
                r"показания\s+для\s+(?:плановой|экстренной)?\s*госпитализац|показания\s+к\s+выписк",
                title,
                flags=re.IGNORECASE,
            )
        )

    def _find_trim_end(self, text: str, matches: list[HeaderMatch]) -> int:
        has_prevention = False
        seen_core_header = False
        seen_core_progress = False
        passed_first_real_block = False
        if len(text) < MIN_TRIM_START * 2:
            min_trim_start = int(len(text) * 0.25)
        else:
            min_trim_start = max(MIN_TRIM_START, int(len(text) * 0.05))
        for header in matches:
            full_title = f"{header.number} {header.title}"
            top_level_number = int(header.number.split(".")[0])
            if top_level_number <= 5:
                seen_core_header = True
                if header.start >= min_trim_start:
                    passed_first_real_block = True
            if top_level_number in {1, 2, 3} and header.start >= min_trim_start:
                seen_core_progress = True
            if self._is_prevention_header(full_title):
                has_prevention = True
                continue
            if not self._is_stop_header(full_title):
                continue
            if header.start < min_trim_start:
                continue
            if (has_prevention and seen_core_progress) or (
                seen_core_header
                and seen_core_progress
                and passed_first_real_block
                and top_level_number >= 6
            ):
                return header.start
        return len(text)

    def _cyrillic_ratio(self, text: str) -> float:
        letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
        if not letters:
            return 0.0
        cyrillic = re.findall(r"[А-Яа-яЁё]", text)
        return len(cyrillic) / len(letters)

    def _is_text_corrupted(self, text: str) -> bool:
        letters_count = len(re.findall(r"[A-Za-zА-Яа-яЁё]", text))
        if letters_count < MIN_LETTERS_FOR_ENCODING_CHECK:
            return False
        return self._cyrillic_ratio(text) < CYRILLIC_RATIO_FLOOR

    def _split_large_section(self, text: str, parent_number: str) -> list[str]:
        if len(text) <= MAX_SECTION_LENGTH:
            return [text]
        sub_parts = self._split_by_subheaders(text, parent_number)
        if not sub_parts:
            return self._split_by_paragraphs(text)

        chunks: list[str] = []
        for part in sub_parts:
            if len(part) <= MAX_SECTION_LENGTH:
                chunks.append(part)
                continue
            chunks.extend(self._split_by_paragraphs(part))
        return [chunk for chunk in chunks if chunk.strip()]

    def _split_by_subheaders(self, text: str, parent_number: str) -> list[str]:
        parent = parent_number.rstrip(".")
        subheader_re = re.compile(
            rf"(?m)^\s*{re.escape(parent)}\.\d+(?:\.\d+)*\.?\s+[^\n]{{3,260}}\s*$"
        )
        matches = list(subheader_re.finditer(text))
        if len(matches) < 2:
            return []

        segments: list[str] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i < len(matches) - 1 else len(text)
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
        return segments

    def _split_by_paragraphs(self, text: str) -> list[str]:
        paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]
        if not paragraphs:
            return self._hard_split(text, MAX_SECTION_LENGTH)

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(paragraph) > MAX_SECTION_LENGTH:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_large_paragraph(paragraph))
                continue
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= MAX_SECTION_LENGTH:
                current = candidate
                continue
            chunks.append(current.strip())
            current = paragraph
        if current:
            chunks.append(current.strip())
        return chunks

    def _split_large_paragraph(self, paragraph: str) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        if not sentences:
            return self._hard_split(paragraph, MAX_SECTION_LENGTH)

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > MAX_SECTION_LENGTH:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._hard_split(sentence, MAX_SECTION_LENGTH))
                continue
            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= MAX_SECTION_LENGTH:
                current = candidate
                continue
            chunks.append(current.strip())
            current = sentence
        if current:
            chunks.append(current.strip())
        return chunks

    def _hard_split(self, text: str, max_length: int) -> list[str]:
        return [
            text[i : i + max_length].strip()
            for i in range(0, len(text), max_length)
            if text[i : i + max_length].strip()
        ]
