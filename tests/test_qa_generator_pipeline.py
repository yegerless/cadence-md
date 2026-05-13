"""Pipeline tests for QADatasetGenerator, JSONL I/O, and main entrypoint."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cadence_md.app.pdf_parser.parser import ClinicalSection
from qa_dataset_generator.generator import QADatasetGenerator
from qa_dataset_generator.main import generate_qa_dataset
from qa_dataset_generator.schemas import GenerationPipelineStats, QAPair
from tests.conftest import QA_LONG_CONTENT_MIN_LEN, make_clinical_section


def _valid_qa_pair_for_section(section: ClinicalSection) -> QAPair:
    return QAPair(
        question="Независимый клинический вопрос про тактику и длина ок?",
        answer="Ответ основан на разделе и имеет достаточную длину для валидации.",
        question_type="simple",
        section_type=section.section_type,
        document_title=section.document_title,
        section_title=section.section_title,
        mkb_codes=section.mkb_codes,
        context="Цитата из клинического фрагмента для подтверждения ответа.",
        section_id=section.section_id,
    )


def test_qadataset_generator_rejects_non_positive_sections_per_pdf() -> None:
    with pytest.raises(ValueError, match="sections_per_pdf must be greater than 0"):
        QADatasetGenerator(sections_per_pdf=0)


def test_qadataset_generator_rejects_negative_min_context() -> None:
    with pytest.raises(ValueError, match="min_context_length must be >= 0"):
        QADatasetGenerator(min_context_length=-1)


def test_qadataset_generator_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("qa_dataset_generator.generator.GIGACHAT_API_KEY", "")
    with pytest.raises(ValueError, match="GIGACHAT_API_KEY is not set"):
        QADatasetGenerator()


def test_select_random_sections_skips_source_with_only_short_content(
    gigachat_dummy_key,
    mock_gigachat_class: MagicMock,
) -> None:
    gen = QADatasetGenerator(seed=1, min_context_length=500, sections_per_pdf=3)
    short = ClinicalSection(
        filename="only_short.pdf",
        document_title="T",
        section_type="treatment",
        section_title="A",
        content="x" * 100,
        mkb_codes=[],
    )
    selected, stats = gen._select_random_sections([short])
    assert selected == []
    assert stats.sources_skipped_short == 1
    assert "only_short.pdf" in stats.skipped_sources


def test_select_random_sections_groups_blank_meta_under_single_source(
    gigachat_dummy_key,
    mock_gigachat_class: MagicMock,
) -> None:
    gen = QADatasetGenerator(seed=1, min_context_length=500, sections_per_pdf=1)
    a = make_clinical_section(
        filename="",
        document_title="",
        section_type="treatment",
        section_title="T1",
    )
    b = make_clinical_section(
        filename="",
        document_title="",
        section_type="diagnosis",
        section_title="T2",
    )
    selected, stats = gen._select_random_sections([a, b])
    assert stats.sources_total == 1
    assert len(selected) == 1


def test_select_random_sections_takes_all_long_when_below_cap(
    gigachat_dummy_key,
    mock_gigachat_class: MagicMock,
) -> None:
    gen = QADatasetGenerator(seed=1, min_context_length=500, sections_per_pdf=5)
    sections = [
        make_clinical_section(
            filename="one.pdf",
            section_type="treatment",
            section_title=f"T{i}",
            content="а" * QA_LONG_CONTENT_MIN_LEN,
        )
        for i in range(2)
    ]
    selected, _stats = gen._select_random_sections(sections)
    assert len(selected) == 2


def test_select_random_sections_deterministic_with_seed(
    gigachat_dummy_key,
    mock_gigachat_class: MagicMock,
) -> None:
    def build_pool() -> list[ClinicalSection]:
        rows: list[ClinicalSection] = []
        for i in range(3):
            for st in ("treatment", "diagnosis", "symptoms"):
                rows.append(
                    make_clinical_section(
                        filename="big.pdf",
                        document_title="Doc",
                        section_type=st,
                        section_title=f"{st}-{i}",
                        content="б" * QA_LONG_CONTENT_MIN_LEN,
                    )
                )
        return rows

    pool = build_pool()
    gen_a = QADatasetGenerator(seed=42, min_context_length=500, sections_per_pdf=3)
    gen_b = QADatasetGenerator(seed=42, min_context_length=500, sections_per_pdf=3)
    sel_a, _ = gen_a._select_random_sections(pool)
    sel_b, _ = gen_b._select_random_sections(pool)
    assert [s.section_title for s in sel_a] == [s.section_title for s in sel_b]


def test_append_pairs_writes_jsonl_lines(
    tmp_path: Path,
    gigachat_dummy_key,
    mock_gigachat_class: MagicMock,
) -> None:
    gen = QADatasetGenerator(seed=1)
    out = tmp_path / "pairs.jsonl"
    sec = make_clinical_section()
    pairs = [
        _valid_qa_pair_for_section(sec),
        _valid_qa_pair_for_section(make_clinical_section(section_title="Other")),
    ]
    gen._append_pairs(pairs, out)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert "question" in row and "section_id" in row


def test_generate_from_sections_file_raises_when_output_exists(
    tmp_path: Path,
    gigachat_dummy_key,
    mock_gigachat_class: MagicMock,
) -> None:
    gen = QADatasetGenerator(seed=1)
    sections_file = tmp_path / "in.jsonl"
    sections_file.write_text("", encoding="utf-8")
    output_file = tmp_path / "out.jsonl"
    output_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Output file already exists"):
        gen.generate_from_sections_file(sections_file, output_file)


def test_generate_from_sections_file_skips_invalid_jsonl(
    tmp_path: Path,
    gigachat_dummy_key,
    mock_gigachat_class: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTqdm:
        def __init__(self, iterable, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, **kwargs) -> None:
            return None

    monkeypatch.setattr("qa_dataset_generator.generator.tqdm", _FakeTqdm)
    gen = QADatasetGenerator(seed=1)
    sec = make_clinical_section()
    sections_file = tmp_path / "in.jsonl"
    sections_file.write_text(
        "not-json-at-all\n" + json.dumps(sec.to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_file = tmp_path / "out.jsonl"
    pair = _valid_qa_pair_for_section(sec)

    def _run(_section: ClinicalSection) -> QAPair:
        return pair

    monkeypatch.setattr(gen.graph, "run", _run)
    _pairs, stats = gen.generate_from_sections_file(sections_file, output_file)
    assert stats.skipped_invalid_jsonl == 1
    assert len(_pairs) == 1
    assert output_file.is_file()


def test_generate_qa_dataset_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sections_file = tmp_path / "sections.jsonl"
    sections_file.write_text("", encoding="utf-8")
    output_file = tmp_path / "qa.jsonl"
    stats = GenerationPipelineStats(
        sections_loaded=1,
        sections_selected=1,
        skipped_invalid_jsonl=0,
        sections_processed=1,
        pairs_generated=1,
        sections_failed=0,
        min_context_length=500,
        sources_total=1,
        sources_skipped_short=0,
        skipped_sources=(),
        sections_filtered_short=0,
    )
    sec = make_clinical_section()
    sample_pair = _valid_qa_pair_for_section(sec)

    mock_instance = MagicMock()
    mock_instance.generate_from_sections_file.return_value = ([sample_pair], stats)
    monkeypatch.setattr("qa_dataset_generator.main.QADatasetGenerator", lambda **_: mock_instance)

    generate_qa_dataset(
        sections_file=sections_file,
        output_file=output_file,
        model="GigaChat-2-Max",
        temperature=0.0,
        max_context=10_000,
        sections_per_pdf=3,
        seed=None,
    )
    report_path = tmp_path / "qa_generation_report.md"
    assert report_path.is_file()
    assert "# QA dataset generation report" in report_path.read_text(encoding="utf-8")
    mock_instance.generate_from_sections_file.assert_called_once()
