from cadence_md.app.pdf_parser.parser import (
    MAX_SECTION_LENGTH,
    ClinicalGuidelinesParser,
    ClinicalSection,
)
from cadence_md.app.pdf_parser.text_extraction import ExtractionResult, TextQuality


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
