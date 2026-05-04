from pathlib import Path

import pytest

from cadence_md.app.pdf_parser.parser import (
    MAX_SECTION_LENGTH,
    ClinicalGuidelinesParser,
    ClinicalSection,
    HeaderMatch,
)
from cadence_md.app.pdf_parser.text_extraction import ExtractionResult, TextQuality


def _dummy_extraction(text: str, layout_headers: list[str] | None = None) -> ExtractionResult:
    """Build a synthetic high-quality ExtractionResult for tests."""
    return ExtractionResult(
        text=text,
        total_pages=1,
        mode_used="dummy",
        quality=TextQuality(avg_chars_per_page=1000.0, empty_page_ratio=0.0, score=1000.0),
        layout_headers=layout_headers or [],
    )


def test_clinical_section_builds_stable_section_id() -> None:
    section = ClinicalSection(
        filename="guideline.pdf",
        document_title="Клинические рекомендации",
        section_type="treatment",
        section_title="3. Лечение",
        content="Текст раздела",
        mkb_codes=["A00"],
    )
    same_section = ClinicalSection.from_dict(
        {
            "filename": "guideline.pdf",
            "document_title": "Клинические рекомендации",
            "section_type": "treatment",
            "section_title": "3. Лечение",
            "content": "Другой текст не влияет на id",
            "mkb_codes": ["A00"],
        }
    )
    other_section = ClinicalSection(
        filename="guideline.pdf",
        document_title="Клинические рекомендации",
        section_type="treatment",
        section_title="3.1 Хирургическое лечение",
        content="Текст раздела",
        mkb_codes=["A00"],
    )

    assert section.section_id == same_section.section_id
    assert section.section_id != other_section.section_id
    assert section.to_dict()["section_id"] == section.section_id


def test_extract_sections_skips_toc_and_trims_tail() -> None:
    parser = ClinicalGuidelinesParser()
    toc = (
        "1. Краткая информация ................ 3\n"
        "1.1 Определение ................ 4\n"
        "1.2 Этиология и патогенез ................ 5\n"
        "2. Диагностика ................ 6\n"
        "3. Лечение ................ 8\n"
        "4. Реабилитация ................ 10\n"
        "5. Профилактика ................ 12\n"
        "6. Организация оказания медицинской помощи ................ 13"
    )
    body = (
        "1. Краткая информация по заболеванию или состоянию\nТекст раздела 1.\n\n"
        "2. Диагностика заболевания или состояния\nТекст диагностики.\n\n"
        "3. Лечение, включая медикаментозную или немедикаментозную терапии\nТекст лечения.\n\n"
        "4. Медицинская реабилитация и санаторно-курортное лечение\nТекст реабилитации.\n\n"
        "5. Профилактика и диспансерное наблюдение\nТекст профилактики.\n\n"
        "6. Организация оказания медицинской помощи\nЭто должно быть отрезано."
    )
    sections = parser._extract_sections(f"СОДЕРЖАНИЕ\n{toc}\n\n{body}")

    assert sections, "Парсер должен найти секции в теле документа"
    joined_titles = [f"{number} {title}" for number, title, _ in sections]
    assert not any("Организация оказания" in title for title in joined_titles)
    assert any("Профилактика и диспансерное наблюдение" in title for title in joined_titles)


def test_split_large_section_uses_subheaders() -> None:
    parser = ClinicalGuidelinesParser()
    large_text = (
        "3.1 Консервативное лечение\n"
        + ("А" * 5300)
        + "\n\n3.2 Хирургическое лечение\n"
        + ("Б" * 5300)
    )

    chunks = parser._split_large_section(large_text, "3")

    assert len(chunks) == 2
    assert chunks[0].startswith("3.1")
    assert chunks[1].startswith("3.2")
    assert all(len(chunk) <= MAX_SECTION_LENGTH for chunk in chunks)


def test_split_large_section_fallback_to_paragraphs() -> None:
    parser = ClinicalGuidelinesParser()
    paragraph = " ".join(["абзац"] * 1200)
    text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}"

    chunks = parser._split_large_section(text, "2")

    assert len(chunks) >= 2
    assert all(len(chunk) <= MAX_SECTION_LENGTH for chunk in chunks)


def test_normalize_header_text_handles_common_typos() -> None:
    parser = ClinicalGuidelinesParser()
    normalized = parser._normalize_header_text("4. Медицинская реабилитауия и санаторно-курортное")

    assert "реабилитац" in normalized.lower()


def test_classify_section_matches_core_titles() -> None:
    parser = ClinicalGuidelinesParser()

    assert parser._classify_section("2. Диагностика заболевания или состояния") == "diagnosis"
    assert parser._classify_section("3. Консервативное лечение") == "treatment"
    assert parser._classify_section("5. Профилактика и диспансерное наблюдение") == "prevention"
    assert parser._classify_section("5. ПдофЕлзктнЕа г. диспансерное наблюдение") == "prevention"


def test_is_likely_header_rejects_numeric_list_items() -> None:
    parser = ClinicalGuidelinesParser()

    assert not parser._is_likely_header("1", "Гнойно-воспалительные заболевания кожи промежности")
    assert not parser._is_likely_header("10", "Осложнения лучевой терапии.")
    assert not parser._is_likely_header(
        "8", "Комментарии: Клиническая диагностика порока в обычных случаях достаточно проста"
    )
    assert not parser._is_likely_header("3", "; а ” функциональной диагностики")
    assert not parser._is_likely_header(
        "1", "Климзтолечение(воздушные, солнечные ванны, морские купания) рекомендуется"
    )
    assert not parser._is_likely_header("5", "находиться на диспансерном наблюдении в течение года")


def test_should_skip_section_title_for_epidemiology() -> None:
    parser = ClinicalGuidelinesParser()

    assert parser._should_skip_section_title("1.3 Эпидемиология заболевания или состояния")
    assert parser._should_skip_section_title(
        "1.4 Особенности кодирования заболевания или состояния"
    )
    assert parser._should_skip_section_title(
        "1.4 Особенности кодирования заболевания илисостояния "
        "(группы заболеваний или состояний) "
        "поМеждународной статистической классификацииболезней и проблем, связанных со здоровьем"
    )
    assert parser._should_skip_section_title("7.2 Стратификация риска хирургического лечения")
    assert not parser._should_skip_section_title("2. Диагностика заболевания или состояния")


def test_detect_toc_end_uses_early_cluster_not_late_noise() -> None:
    parser = ClinicalGuidelinesParser()
    toc_lines = (
        "1. Краткая информация ................ 5\n"
        "1.1 Определение ................ 6\n"
        "1.2 Этиология и патогенез ................ 6\n"
        "1.3 Эпидемиология ................ 7\n"
        "2. Диагностика ................ 8\n"
        "2.1 Жалобы и анамнез ................ 9\n"
        "3. Лечение ................ 12\n"
        "4. Реабилитация ................ 16\n"
        "5. Профилактика ................ 17"
    )
    noise = (
        "4.5 Теплев Н.В. ................ 120\n"
        "5. Врач ультразвуковой диагностики ................ 121\n"
        "4. Несравнительные исследования ................ 122\n"
        "5. Имеется лишь обоснование механизма действия ................ 123"
    )
    body = (
        "1. Краткая информация по заболеванию или состоянию\n"
        "1.1 Определение заболевания или состояния\n"
        "2. Диагностика заболевания или состояния\n"
        "3. Лечение, включая медикаментозную или немедикаментозную терапии\n"
        "4. Медицинская реабилитация и санаторно-курортное лечение\n"
        "5. Профилактика и диспансерное наблюдение"
    )
    text = f"{toc_lines}\n\n{body}\n\n{noise}"

    toc_end = parser._detect_toc_end(text)
    assert toc_end > 0
    assert toc_end < text.find(body)


def test_extract_header_matches_handles_ocr_number_and_missing_major_number() -> None:
    parser = ClinicalGuidelinesParser()
    text = (
        "3.2 Хирургическое лечение\nТекст лечения.\n\n"
        "Медицинская реабилитация и санаторно-курортное лечение, медицинские показания и "
        "противопоказания к применению методов медицинской реабилитации\n"
        "Текст реабилитации.\n\n"
        "5, ПдофЕлзктнЕа г. диспансерное наблюдение, медицинские показания и противопоказания\n"
        "Текст профилактики.\n"
    )
    normalized = parser._normalize_extracted_text(text)
    headers = parser._extract_header_matches(normalized)
    titles = [f"{h.number} {h.title.lower()}" for h in headers]

    assert any(t.startswith("4 медицинск") for t in titles)
    assert any("диспансер" in t and t.startswith("5 ") for t in titles)


def test_merge_layout_headers_adds_missing_candidates() -> None:
    parser = ClinicalGuidelinesParser()
    text = (
        "1. Краткая информация по заболеванию или состоянию\nТекст.\n\n"
        "3. Лечение, включая медикаментозную терапии\nТекст лечения."
    )
    regex_headers = parser._extract_header_matches(text)
    merged = parser._merge_layout_headers(
        regex_headers, text, ["2. Диагностика заболевания или состояния"]
    )

    titles = [f"{item.number} {item.title}" for item in merged]
    assert any(title.startswith("1 Краткая информация") for title in titles)


def test_parse_pdf_uses_custom_extractor(tmp_path) -> None:
    class DummyExtractor:
        def extract(self, _pdf_path):
            long_body = " ".join(["диагностика"] * 30)
            return ExtractionResult(
                text=(
                    f"1. Краткая информация по заболеванию или состоянию\n{long_body}.\n\n"
                    f"2. Диагностика заболевания или состояния\n{long_body}.\n\n"
                    f"3. Лечение, включая медикаментозную терапии\n{long_body}."
                ),
                total_pages=1,
                mode_used="dummy",
                quality=TextQuality(avg_chars_per_page=1000.0, empty_page_ratio=0.0, score=1000.0),
                layout_headers=[],
            )

    pdf_file = tmp_path / "dummy.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%dummy")
    parser = ClinicalGuidelinesParser(extractor=DummyExtractor())
    sections = parser.parse_pdf(pdf_file) or []

    assert sections
    assert any(section.section_type == "diagnosis" for section in sections)


def test_extract_header_matches_handles_inline_concatenated_headers() -> None:
    parser = ClinicalGuidelinesParser()
    text = (
        "1. Краткая информация по заболеванию или состоянию "
        "1.1 Определение заболевания или состояния "
        "1.2 Этиология и патогенез заболевания или состояния\n"
        "Текст раздела.\n"
    )

    normalized = parser._normalize_extracted_text(text)
    headers = parser._extract_header_matches(normalized)
    numbers = [header.number for header in headers]

    assert "1" in numbers
    assert "1.1" in numbers
    assert "1.2" in numbers


def test_find_trim_end_ignores_early_stop_headers() -> None:
    parser = ClinicalGuidelinesParser()
    text = "X" * 20000
    matches = [
        parser._extract_header_matches("4. Медицинская реабилитация")[0],
        parser._extract_header_matches("5. Профилактика и диспансерное наблюдение")[0],
        parser._extract_header_matches("6. Организация оказания медицинской помощи")[0],
        parser._extract_header_matches("2. Диагностика заболевания или состояния")[0],
    ]
    shifted = [
        type(match)(
            number=match.number,
            title=match.title,
            start=1000 + idx * 500,
            end=1100 + idx * 500,
            level=match.level,
        )
        for idx, match in enumerate(matches)
    ]

    trim_end = parser._find_trim_end(text, shifted)

    assert trim_end == len(text)


def test_is_text_corrupted_detects_mojibake_like_text() -> None:
    parser = ClinicalGuidelinesParser()

    assert parser._is_text_corrupted("A" * 400)
    assert not parser._is_text_corrupted("Клинические рекомендации " * 30)


# --- parse_pdf / parse_directory ---


def test_parse_directory_aggregates_pdfs(tmp_path: Path) -> None:
    class TwoChapterExtractor:
        def extract(self, _pdf_path):
            long_body = " ".join(["диагностика"] * 30)
            text = (
                f"1. Краткая информация по заболеванию или состоянию\n{long_body}.\n\n"
                f"2. Диагностика заболевания или состояния\n{long_body}.\n\n"
            )
            return _dummy_extraction(text)

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4\n%dummy")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4\n%dummy")
    parser = ClinicalGuidelinesParser(extractor=TwoChapterExtractor())

    sections = parser.parse_directory(tmp_path)
    filenames = {section.filename for section in sections}

    assert filenames == {"a.pdf", "b.pdf"}
    assert len(sections) >= 4


def test_parse_pdf_returns_none_on_keyerror(tmp_path: Path) -> None:
    class BrokenExtractor:
        def extract(self, _pdf_path):
            raise KeyError("intentional failure")

    pdf_file = tmp_path / "broken.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    parser = ClinicalGuidelinesParser(extractor=BrokenExtractor())

    assert parser.parse_pdf(pdf_file) is None


def test_parse_pdf_inherits_section_type_from_chapter(tmp_path: Path) -> None:
    class ChapterAndSubsectionExtractor:
        def extract(self, _pdf_path):
            body_a = " ".join(["прокладка"] * 30)
            body_b = " ".join(["детали"] * 30)
            text = (
                f"3. Лечение, включая медикаментозную или немедикаментозную терапии\n{body_a}.\n\n"
                f"3.1 Подзаголовок без явных слов о методах\n{body_b}.\n\n"
            )
            return _dummy_extraction(text)

    pdf_file = tmp_path / "treatment.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    parser = ClinicalGuidelinesParser(extractor=ChapterAndSubsectionExtractor())

    sections = parser.parse_pdf(pdf_file) or []
    subsection = [s for s in sections if s.section_title.startswith("3.1")]

    assert subsection, "Подсекция 3.1 должна попасть в результат"
    assert subsection[0].section_type == "treatment"


def test_parse_pdf_chunks_carry_part_suffix(tmp_path: Path) -> None:
    class LongChapterExtractor:
        def extract(self, _pdf_path):
            sentence = "Лечение пациентов и медикаментозная терапия для всех групп. "
            large_body = sentence * 600
            text = (
                f"3. Лечение, включая медикаментозную или немедикаментозную терапии\n{large_body}"
            )
            return _dummy_extraction(text)

    pdf_file = tmp_path / "long.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    parser = ClinicalGuidelinesParser(extractor=LongChapterExtractor())

    sections = parser.parse_pdf(pdf_file) or []

    assert len(sections) >= 2
    assert all("[part" in section.section_title for section in sections)
    assert all(len(section.content) <= MAX_SECTION_LENGTH for section in sections)


# --- Document metadata ---


def test_extract_document_title_from_header() -> None:
    parser = ClinicalGuidelinesParser()
    text = "Минздрав Российской Федерации\nКлинические рекомендации\nЯзва желудка\nМКБ-10 K25\n"

    title = parser._extract_document_title(text, Path("dummy.pdf"))

    assert title == "Язва желудка"


def test_extract_document_title_falls_back_to_filename() -> None:
    parser = ClinicalGuidelinesParser()
    text = "Текст без характерных меток"

    title = parser._extract_document_title(text, Path("kr_diabetes_type_2.pdf"))

    assert title == "kr diabetes type 2.pdf"


def test_extract_icd_codes_dedupes_and_only_in_header() -> None:
    parser = ClinicalGuidelinesParser()
    header = (
        "Клинические рекомендации\n"
        "Язва желудка\n"
        "МКБ-10 K25, K25.1, K25, K26\n"
        "Дополнительная информация в шапке.\n"
    )
    padding_size = 3000 - len(header) + 10
    text = header + ("x" * padding_size) + "\nK99 в теле документа должно быть проигнорировано"

    codes = parser._extract_icd_codes(text)

    assert codes == ["K25", "K25.1", "K26"]


# --- Классификация и очистка ---


def test_classify_section_excludes_appendices() -> None:
    parser = ClinicalGuidelinesParser()

    assert parser._classify_section("Приложение А. Алгоритмы ведения пациента") is None
    assert parser._classify_section("Список литературы") is None
    assert parser._classify_section("Критерии оценки качества медицинской помощи") is None


def test_clean_section_text_strips_noise() -> None:
    parser = ClinicalGuidelinesParser()
    text = (
        "Содержание раздела .... 5\n"
        "Подробности обследования см. таблицу 5 ниже.\n"
        "Описание [1] и сравнение [2,3] приведены далее.\n"
        "Также см. рис. 2 для иллюстрации.\n"
        "Картинка Рисунок 3 описана отдельно.\n"
    )

    cleaned = parser._clean_section_text(text)

    assert "...." not in cleaned
    assert "[1]" not in cleaned
    assert "[2,3]" not in cleaned
    assert "таблицу 5" not in cleaned
    assert "Рисунок 3" not in cleaned
    assert "см. рис" not in cleaned.lower()
    assert "Описание" in cleaned


# --- Нормализация текста ---


def test_normalize_extracted_text_joins_hyphenated_words() -> None:
    parser = ClinicalGuidelinesParser()

    out = parser._normalize_extracted_text("диагно-\nстика заболевания")

    assert "диагностика" in out
    assert "диагно-" not in out


def test_normalize_extracted_text_fixes_comma_in_number() -> None:
    parser = ClinicalGuidelinesParser()
    text = "Шапка\n1, Лечение\nТело раздела"

    out = parser._normalize_extracted_text(text)

    assert "1. Лечение" in out


def test_normalize_extracted_text_glues_split_numbered_heading() -> None:
    parser = ClinicalGuidelinesParser()
    text = "3.1\nЛечение пациентов\nТело"

    out = parser._normalize_extracted_text(text)

    assert "3.1 Лечение" in out


# --- Header detection ---


def test_extract_inline_header_matches_requires_two_anchors() -> None:
    parser = ClinicalGuidelinesParser()
    text = "1.1 Какой-то длинный заголовок без второго anchor на этой строке"

    assert parser._extract_inline_header_matches(text) == []


def test_extract_unnumbered_major_headers_synthesizes_4_and_5() -> None:
    parser = ClinicalGuidelinesParser()
    text = (
        "Какой-то предшествующий текст\n\n"
        "Медицинская реабилитация и санаторно-курортное лечение пациентов\n\n"
        "Промежуточный текст между разделами длиной не менее ста символов "
        "для большей реалистичности теста.\n\n"
        "Диспансерное наблюдение пациентов и контроль состояния после лечения\n\n"
    )

    headers = parser._extract_unnumbered_major_headers(text, [])
    numbers = [header.number for header in headers]

    assert "4" in numbers
    assert "5" in numbers


def test_has_nearby_same_top_header_suppresses_duplicate() -> None:
    parser = ClinicalGuidelinesParser()
    text = "Шапка\n\nМедицинская реабилитация и санаторно-курортное лечение пациентов\n\n"
    target_pos = text.find("Медицинская реабилитация")
    existing = [
        HeaderMatch(
            number="4",
            title="Медицинская реабилитация",
            start=target_pos,
            end=target_pos + 25,
            level=1,
        )
    ]

    headers = parser._extract_unnumbered_major_headers(text, existing)

    assert headers == []


# --- Утилиты ---


def test_range_overlaps_basic() -> None:
    parser = ClinicalGuidelinesParser()

    assert parser._range_overlaps(0, 5, [(3, 8)])
    assert parser._range_overlaps(0, 5, [(5, 10)])
    assert parser._range_overlaps(5, 10, [(0, 5)])
    assert not parser._range_overlaps(0, 5, [(10, 15)])
    assert not parser._range_overlaps(0, 5, [])


def test_header_level_and_top_level_number() -> None:
    parser = ClinicalGuidelinesParser()

    assert parser._header_level("1") == 1
    assert parser._header_level("1.2") == 2
    assert parser._header_level("1.2.3") == 3

    assert parser._top_level_number("1") == 1
    assert parser._top_level_number("3.4") == 3
    assert parser._top_level_number("abc") is None


def test_is_plausible_top_level_title_anchors() -> None:
    parser = ClinicalGuidelinesParser()

    assert parser._is_plausible_top_level_title("диагностика заболевания")
    assert parser._is_plausible_top_level_title("лечение пациентов")
    assert parser._is_plausible_top_level_title("диспансерное наблюдение пациентов")
    assert not parser._is_plausible_top_level_title("комментарии экспертов")


def test_is_prevention_header_and_is_stop_header() -> None:
    parser = ClinicalGuidelinesParser()

    assert parser._is_prevention_header("5. Профилактика и диспансерное наблюдение")
    assert parser._is_prevention_header("4.1 Диспансерное наблюдение")
    assert not parser._is_prevention_header("3. Лечение")

    assert parser._is_stop_header("Список литературы")
    assert parser._is_stop_header("Приложение А. Алгоритмы")
    assert parser._is_stop_header("Показания для плановой госпитализации")
    assert parser._is_stop_header("Показания к выписке пациента")
    assert not parser._is_stop_header("3. Лечение")


# --- TOC detection ---


def test_detect_toc_end_returns_zero_without_signals() -> None:
    parser = ClinicalGuidelinesParser()
    text = "Просто текст без TOC и без заголовков с дотами.\n" * 5

    assert parser._detect_toc_end(text) == 0


def test_detect_toc_end_uses_second_top_level_one_fallback() -> None:
    parser = ClinicalGuidelinesParser()
    toc_lines = [
        "1 Краткая информация",
        "1.1 Определение заболевания или состояния",
        "1.2 Этиология и патогенез заболевания",
        "1.3 Эпидемиология заболевания",
        "1.4 Особенности кодирования",
        "1.5 Классификация заболевания",
        "2 Диагностика",
        "2.1 Жалобы и анамнез",
        "2.2 Физикальное обследование пациента",
        "2.3 Лабораторные диагностические исследования",
        "2.4 Инструментальные диагностические исследования",
        "3 Лечение пациентов",
        "3.1 Консервативное лечение",
        "4 Медицинская реабилитация",
        "5 Профилактика и диспансерное наблюдение",
    ]
    body_lines = [
        "1 Краткая информация по заболеванию или состоянию",
        "Текст краткой информации.",
        "2 Диагностика заболевания или состояния",
        "Текст диагностики.",
        "3 Лечение пациентов и медикаментозная терапия",
        "Текст лечения.",
        "4 Медицинская реабилитация и санаторно-курортное лечение",
        "Текст реабилитации.",
        "5 Профилактика и диспансерное наблюдение",
        "Текст профилактики.",
    ]
    text = "СОДЕРЖАНИЕ\n" + "\n".join(toc_lines) + "\n" + "\n".join(body_lines)
    body_one_offset = text.find("1 Краткая информация по заболеванию")

    toc_end = parser._detect_toc_end(text)

    assert toc_end == body_one_offset


# --- Splitting ---


def test_split_by_subheaders_needs_two_matches() -> None:
    parser = ClinicalGuidelinesParser()
    text = "3.1 Подзаголовок\nТекст подзаголовка раздела"

    assert parser._split_by_subheaders(text, "3") == []


def test_split_by_paragraphs_uses_hard_split_for_huge_unsplittable_paragraph() -> None:
    parser = ClinicalGuidelinesParser()
    text = "А" * (MAX_SECTION_LENGTH + 5000)

    chunks = parser._split_by_paragraphs(text)

    assert len(chunks) >= 2
    assert all(len(chunk) <= MAX_SECTION_LENGTH for chunk in chunks)


def test_split_large_paragraph_uses_sentence_boundaries() -> None:
    parser = ClinicalGuidelinesParser()
    sentence_a = "А" * 4500 + "."
    sentence_b = "Б" * 4500 + "."
    sentence_c = "В" * 4500 + "."
    paragraph = f"{sentence_a} {sentence_b} {sentence_c}"

    chunks = parser._split_large_paragraph(paragraph)

    assert len(chunks) == 2
    assert all(len(chunk) <= MAX_SECTION_LENGTH for chunk in chunks)


def test_hard_split_returns_fixed_windows() -> None:
    parser = ClinicalGuidelinesParser()
    max_length = 100
    text = "А" * 250

    chunks = parser._hard_split(text, max_length)

    assert len(chunks) == 3
    assert all(len(chunk) <= max_length for chunk in chunks)


# --- Encoding / variant choice ---


def test_cyrillic_ratio_empty_string() -> None:
    parser = ClinicalGuidelinesParser()

    assert parser._cyrillic_ratio("") == 0.0
    assert parser._cyrillic_ratio("12345 !@#") == 0.0


def test_is_better_text_variant_corruption_tiebreak() -> None:
    parser = ClinicalGuidelinesParser()
    current = "A" * 400
    candidate = "Клинические рекомендации " * 30

    assert parser._is_better_text_variant(
        current_text=current,
        current_score=10.0,
        candidate_text=candidate,
        candidate_score=10.0,
    )


def test_is_better_text_variant_uses_cyrillic_ratio() -> None:
    parser = ClinicalGuidelinesParser()
    current = "Клинические рекомендации " * 5 + "abcdef ghij " * 30
    candidate = "Клинические рекомендации " * 30

    assert parser._is_better_text_variant(
        current_text=current,
        current_score=100.0,
        candidate_text=candidate,
        candidate_score=100.0,
    )


def test_try_extract_native_variant_returns_none_without_method() -> None:
    class MinimalExtractor:
        def extract(self, _pdf_path):
            return _dummy_extraction("текст")

    parser = ClinicalGuidelinesParser(extractor=MinimalExtractor())

    assert parser._try_extract_native_variant(Path("dummy.pdf")) is None


def test_extract_text_with_page_markers_uses_native_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class FallbackExtractor:
        def extract(self, _pdf_path):
            return ExtractionResult(
                text="A" * 600,
                total_pages=2,
                mode_used="layout",
                quality=TextQuality(avg_chars_per_page=300.0, empty_page_ratio=0.0, score=10.0),
                layout_headers=["fake header"],
            )

        def _extract_native(self, _pdf_path):
            return ("Клинические рекомендации по заболеванию " * 30, 2)

        def _assess_quality(self, text: str, pages: int) -> TextQuality:
            avg = len(text) / max(pages, 1)
            return TextQuality(avg_chars_per_page=avg, empty_page_ratio=0.0, score=avg)

    pdf_file = tmp_path / "fallback.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    parser = ClinicalGuidelinesParser(extractor=FallbackExtractor())

    with caplog.at_level("WARNING"):
        text, pages, headers = parser._extract_text_with_page_markers(pdf_file)

    assert "Клинические" in text
    assert "A" * 600 not in text
    assert pages == 2
    assert headers == []
