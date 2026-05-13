"""Chunk metadata attached after splitting clinical sections for Qdrant indexing."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from cadence_md.app.pdf_parser.parser import ClinicalSection
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.settings import ChunkConfig


def _manager_with_chunking(cfg: ChunkConfig) -> QdrantManager:
    mgr = object.__new__(QdrantManager)
    mgr.chunking_cfg = cfg
    mgr.data_dir = Path("data/main_specialities")
    return mgr  # type: ignore[return-value]


def test_create_chunks_sets_chunk_metadata_for_split_section() -> None:
    cfg = ChunkConfig(chunk_size=80, chunk_overlap=10)
    mgr = _manager_with_chunking(cfg)
    section = ClinicalSection(
        filename="rec.pdf",
        document_title="Клинические рекомендации",
        section_type="treatment",
        section_title="Лечение",
        content=("Краткий текст. " * 50).strip(),
        mkb_codes=["I10"],
    )
    sid = section.section_id

    create_chunks = mgr._QdrantManager__create_chunks
    chunks = create_chunks([section])

    assert len(chunks) >= 2
    for i, ch in enumerate(chunks):
        meta = ch.metadata
        assert meta["chunk_index"] == i
        assert meta["chunk_total"] == len(chunks)
        assert meta["chunk_size"] == cfg.chunk_size
        assert meta["chunk_overlap"] == cfg.chunk_overlap
        assert meta["chunk_id"] == f"{sid}_chunk_{i:04d}"
        assert meta["filename"] == "rec.pdf"
        assert meta["document_title"] == "Клинические рекомендации"
        assert meta["section_title"] == "Лечение"
        assert meta["section_id"] == sid
        assert meta["source_path"] == "main_specialities/rec.pdf"


def test_attach_chunk_metadata_fallback_prefix_without_section_id() -> None:
    cfg = ChunkConfig(chunk_size=512, chunk_overlap=64)
    mgr = _manager_with_chunking(cfg)
    docs = [
        Document(
            page_content="first",
            metadata={"filename": "x.pdf", "section_title": "A", "document_title": "D"},
        ),
        Document(
            page_content="second",
            metadata={"filename": "x.pdf", "section_title": "A", "document_title": "D"},
        ),
    ]

    out = mgr._attach_chunk_metadata(docs)

    assert out is docs
    assert docs[0].metadata["chunk_index"] == 0
    assert docs[1].metadata["chunk_index"] == 1
    assert docs[0].metadata["chunk_total"] == 2
    assert docs[1].metadata["chunk_total"] == 2
    pid0 = docs[0].metadata["chunk_id"]
    pid1 = docs[1].metadata["chunk_id"]
    assert pid0.startswith("fallback_")
    assert pid1.startswith("fallback_")
    assert pid0.endswith("_chunk_0000")
    assert pid1.endswith("_chunk_0001")
    assert pid0[: len("fallback_") + 24] == pid1[: len("fallback_") + 24]
