"""Tests for cadence_md.app.pdf_parser.text_extraction.HybridTextExtractor.

The extractor wraps pypdf (native) and PyMuPDF/fitz (layout). To avoid touching
real PDF files we monkeypatch pypdf.PdfReader, fitz.open, or the private
_load_pymupdf / _extract_native / _extract_layout methods directly.
"""

from pathlib import Path
from typing import Any

import pytest

from cadence_md.app.pdf_parser import text_extraction
from cadence_md.app.pdf_parser.text_extraction import (
    ExtractionResult,
    HybridTextExtractor,
    TextQuality,
)

# --- Helpers for fitz/pypdf fakes ---


def _make_span(
    text: str,
    *,
    size: float = 10.0,
    font: str = "Regular",
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> dict[str, Any]:
    return {"text": text, "size": size, "font": font, "bbox": list(bbox)}


def _make_line(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {"spans": spans}


def _make_block(lines: list[dict[str, Any]], *, block_type: int = 0) -> dict[str, Any]:
    return {"type": block_type, "lines": lines}


class _FakePage:
    def __init__(self, blocks: list[dict[str, Any]]) -> None:
        self._data = {"blocks": blocks}

    def get_text(self, _kind: str) -> dict[str, Any]:
        return self._data


class _FakeDoc:
    def __init__(self, pages: list[_FakePage]) -> None:
        self._pages = pages
        self.closed = False

    def __iter__(self):
        return iter(self._pages)

    def __len__(self) -> int:
        return len(self._pages)

    def close(self) -> None:
        self.closed = True


class _FakeFitz:
    def __init__(self, doc: _FakeDoc) -> None:
        self._doc = doc
        self.calls: list[str] = []

    def open(self, path: str) -> _FakeDoc:
        self.calls.append(path)
        return self._doc


class _FakePypdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePypdfReader:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.pages = [
            _FakePypdfPage("Текст страницы 1 с информацией о лечении"),
            _FakePypdfPage("Текст страницы 2 с диагностикой"),
        ]


# --- _assess_quality ---


def test_assess_quality_empty_text() -> None:
    extractor = HybridTextExtractor()

    quality = extractor._assess_quality("", pages_count=3)

    assert quality.avg_chars_per_page == 0.0
    assert quality.empty_page_ratio == 1.0
    assert quality.score == 0.0


def test_assess_quality_normal_text() -> None:
    extractor = HybridTextExtractor()
    text = "page1\npage2\n"

    quality = extractor._assess_quality(text, pages_count=2)

    assert quality.avg_chars_per_page == pytest.approx(len(text) / 2)
    assert quality.empty_page_ratio == pytest.approx(1 - 2 / 3)
    expected_score = quality.avg_chars_per_page * (1 - quality.empty_page_ratio)
    assert quality.score == pytest.approx(expected_score)


def test_assess_quality_handles_zero_pages_count() -> None:
    extractor = HybridTextExtractor()

    quality = extractor._assess_quality("Какой-то текст", pages_count=0)

    assert quality.avg_chars_per_page == pytest.approx(len("Какой-то текст"))


# --- _is_quality_poor ---


def test_is_quality_poor_low_avg_chars() -> None:
    extractor = HybridTextExtractor()
    quality = TextQuality(avg_chars_per_page=500.0, empty_page_ratio=0.1, score=450.0)

    assert extractor._is_quality_poor(quality)


def test_is_quality_poor_high_empty_ratio() -> None:
    extractor = HybridTextExtractor()
    quality = TextQuality(avg_chars_per_page=1500.0, empty_page_ratio=0.6, score=600.0)

    assert extractor._is_quality_poor(quality)


def test_is_quality_poor_passes_thresholds() -> None:
    extractor = HybridTextExtractor()
    quality = TextQuality(avg_chars_per_page=1500.0, empty_page_ratio=0.1, score=1350.0)

    assert not extractor._is_quality_poor(quality)


# --- _looks_like_numbered_header ---


def test_looks_like_numbered_header_positive() -> None:
    extractor = HybridTextExtractor()

    assert extractor._looks_like_numbered_header("1 Краткая информация")
    assert extractor._looks_like_numbered_header("1.2 Лечение")
    assert extractor._looks_like_numbered_header("  3.1.2 Подзаголовок")


def test_looks_like_numbered_header_negative() -> None:
    extractor = HybridTextExtractor()

    assert not extractor._looks_like_numbered_header("Просто текст без номера")
    assert not extractor._looks_like_numbered_header("5)")
    assert not extractor._looks_like_numbered_header("abc 1.2 Лечение")
    assert not extractor._looks_like_numbered_header("")


# --- _load_pymupdf ---


def test_load_pymupdf_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str, *_args: Any, **_kwargs: Any):
        if name == "fitz":
            raise ImportError("fitz not installed in test env")
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(text_extraction.importlib, "import_module", fake_import)

    extractor = HybridTextExtractor()

    assert extractor._load_pymupdf() is None


# --- _extract_native ---


def test_extract_native_uses_pypdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(text_extraction.pypdf, "PdfReader", _FakePypdfReader)
    pdf_file = tmp_path / "x.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    result = HybridTextExtractor()._extract_native(pdf_file)

    assert isinstance(result, ExtractionResult)
    assert "Текст страницы 1" in result.text
    assert "Текст страницы 2" in result.text
    assert result.total_pages == 2
    assert result.mode_used == "native"


# --- _extract_layout ---


def test_extract_layout_returns_none_without_fitz(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(HybridTextExtractor, "_load_pymupdf", lambda self: None)
    pdf_file = tmp_path / "x.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    assert HybridTextExtractor()._extract_layout(pdf_file) is None


def test_extract_layout_collects_headers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = _make_span("Обычный текст параграфа без номера", size=10.0)
    head_by_size = _make_span("1.2 Лечение", size=14.0)
    head_by_bold = _make_span(
        "2 Диагностика",
        size=10.0,
        font="Helvetica-Bold",
    )
    page = _FakePage(
        [
            _make_block(
                [
                    _make_line([body]),
                    _make_line([head_by_size]),
                    _make_line([head_by_bold]),
                ]
            )
        ]
    )
    fake_fitz = _FakeFitz(_FakeDoc([page]))
    monkeypatch.setattr(HybridTextExtractor, "_load_pymupdf", lambda self: fake_fitz)
    pdf_file = tmp_path / "x.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    result = HybridTextExtractor()._extract_layout(pdf_file)

    assert result is not None
    assert result.mode_used == "layout"
    assert result.total_pages == 1
    assert "1.2 Лечение" in result.layout_headers
    assert "2 Диагностика" in result.layout_headers
    assert "Обычный текст параграфа без номера" in result.text


def test_extract_layout_returns_none_when_open_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingFitz:
        def open(self, _path: str):
            raise RuntimeError("corrupt pdf")

    monkeypatch.setattr(HybridTextExtractor, "_load_pymupdf", lambda self: FailingFitz())
    pdf_file = tmp_path / "x.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    assert HybridTextExtractor()._extract_layout(pdf_file) is None


# --- _extract_layout_page_lines ---


def test_extract_layout_page_lines_skips_non_text_blocks() -> None:
    image_block = _make_block(
        [_make_line([_make_span("Не должен учитываться")])],
        block_type=1,
    )
    text_block = _make_block(
        [_make_line([_make_span("Реальный текст")])],
        block_type=0,
    )
    page = _FakePage([image_block, text_block])
    extractor = HybridTextExtractor()

    lines, _headers = extractor._extract_layout_page_lines(page)

    assert lines == ["Реальный текст"]


def test_extract_layout_page_lines_returns_empty_headers_without_fonts() -> None:
    page = _FakePage([_make_block([_make_line([])])])
    extractor = HybridTextExtractor()

    lines, headers = extractor._extract_layout_page_lines(page)

    assert lines == []
    assert headers == []


# --- HybridTextExtractor.extract ---


def test_extract_hybrid_returns_native_when_quality_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    extractor = HybridTextExtractor()
    good = ExtractionResult(
        text="x" * 2000,
        total_pages=1,
        mode_used="native",
        quality=TextQuality(avg_chars_per_page=1500.0, empty_page_ratio=0.1, score=1350.0),
        layout_headers=[],
    )
    monkeypatch.setattr(HybridTextExtractor, "_extract_native", lambda self, p: good)

    layout_calls = {"count": 0}

    def fake_layout(self: HybridTextExtractor, _p: Path) -> ExtractionResult | None:
        layout_calls["count"] += 1
        return ExtractionResult(
            text="should-not-be-used",
            total_pages=1,
            mode_used="layout",
            quality=TextQuality(avg_chars_per_page=0.0, empty_page_ratio=1.0, score=0.0),
            layout_headers=[],
        )

    monkeypatch.setattr(HybridTextExtractor, "_extract_layout", fake_layout)
    pdf_file = tmp_path / "x.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    result = extractor.extract(pdf_file)

    assert result is good
    assert layout_calls["count"] == 0


def test_extract_hybrid_uses_layout_when_native_poor_and_layout_better(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    poor = ExtractionResult(
        text="x",
        total_pages=1,
        mode_used="native",
        quality=TextQuality(avg_chars_per_page=100.0, empty_page_ratio=0.5, score=50.0),
        layout_headers=[],
    )
    better = ExtractionResult(
        text="y" * 1000,
        total_pages=1,
        mode_used="layout",
        quality=TextQuality(avg_chars_per_page=900.0, empty_page_ratio=0.1, score=810.0),
        layout_headers=[],
    )
    monkeypatch.setattr(HybridTextExtractor, "_extract_native", lambda self, p: poor)
    monkeypatch.setattr(HybridTextExtractor, "_extract_layout", lambda self, p: better)
    pdf_file = tmp_path / "x.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    result = HybridTextExtractor().extract(pdf_file)

    assert result is better


def test_extract_hybrid_keeps_native_when_layout_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    poor = ExtractionResult(
        text="x",
        total_pages=1,
        mode_used="native",
        quality=TextQuality(avg_chars_per_page=100.0, empty_page_ratio=0.5, score=50.0),
        layout_headers=[],
    )
    monkeypatch.setattr(HybridTextExtractor, "_extract_native", lambda self, p: poor)
    monkeypatch.setattr(HybridTextExtractor, "_extract_layout", lambda self, p: None)
    pdf_file = tmp_path / "x.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    result = HybridTextExtractor().extract(pdf_file)

    assert result is poor


def test_extract_hybrid_keeps_native_when_layout_score_lower(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    poor_native = ExtractionResult(
        text="native",
        total_pages=1,
        mode_used="native",
        quality=TextQuality(avg_chars_per_page=100.0, empty_page_ratio=0.5, score=50.0),
        layout_headers=[],
    )
    even_worse_layout = ExtractionResult(
        text="layout",
        total_pages=1,
        mode_used="layout",
        quality=TextQuality(avg_chars_per_page=80.0, empty_page_ratio=0.6, score=32.0),
        layout_headers=[],
    )
    monkeypatch.setattr(HybridTextExtractor, "_extract_native", lambda self, p: poor_native)
    monkeypatch.setattr(HybridTextExtractor, "_extract_layout", lambda self, p: even_worse_layout)
    pdf_file = tmp_path / "x.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    result = HybridTextExtractor().extract(pdf_file)

    assert result is poor_native
