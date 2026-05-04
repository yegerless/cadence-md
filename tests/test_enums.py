"""Unit tests for :mod:`cadence_md.app.enums` and :mod:`qa_dataset_generator.enums`."""

from __future__ import annotations

import json

import pytest

from cadence_md.app.enums import (
    QdrantFusionMethod,
    QdrantVectorType,
    SectionType,
    VectorSearchType,
)
from qa_dataset_generator.enums import QuestionType


@pytest.mark.parametrize(
    ("enum_cls", "name", "value"),
    [
        (QuestionType, "SIMPLE", "simple"),
        (QuestionType, "REASONING", "reasoning"),
        (QuestionType, "MULTI_CONTEXT", "multi_context"),
        (QuestionType, "CONDITIONAL", "conditional"),
        (QuestionType, "COMPARISON", "comparison"),
        (SectionType, "DEFINITION", "definition"),
        (SectionType, "SYMPTOMS", "symptoms"),
        (SectionType, "DIAGNOSIS", "diagnosis"),
        (SectionType, "TREATMENT", "treatment"),
        (SectionType, "PREVENTION", "prevention"),
        (SectionType, "REHABILITATION", "rehabilitation"),
        (QdrantVectorType, "DENSE", "dense"),
        (QdrantVectorType, "SPARSE", "sparse"),
        (QdrantFusionMethod, "RRF", "rrf"),
        (QdrantFusionMethod, "DBSF", "dbsf"),
        (VectorSearchType, "DENSE", "dense"),
        (VectorSearchType, "SPARSE", "sparse"),
        (VectorSearchType, "HYBRID", "hybrid"),
    ],
)
def test_str_enum_member_values(
    enum_cls: type[
        QuestionType | SectionType | QdrantVectorType | QdrantFusionMethod | VectorSearchType
    ],
    name: str,
    value: str,
) -> None:
    member = enum_cls[name]
    assert member.value == value
    assert str(member) == value
    assert member == value
    assert isinstance(member, str)


@pytest.mark.parametrize(
    ("enum_cls", "value"),
    [
        (QuestionType, "simple"),
        (SectionType, "treatment"),
        (QdrantVectorType, "dense"),
        (QdrantFusionMethod, "rrf"),
        (VectorSearchType, "hybrid"),
    ],
)
def test_str_enum_constructible_from_string(
    enum_cls: type[
        QuestionType | SectionType | QdrantVectorType | QdrantFusionMethod | VectorSearchType
    ],
    value: str,
) -> None:
    assert enum_cls(value).value == value


@pytest.mark.parametrize(
    "enum_cls",
    [QuestionType, SectionType, QdrantVectorType, QdrantFusionMethod, VectorSearchType],
)
def test_str_enum_invalid_value_raises(
    enum_cls: type[
        QuestionType | SectionType | QdrantVectorType | QdrantFusionMethod | VectorSearchType
    ],
) -> None:
    with pytest.raises(ValueError):
        enum_cls("not_a_valid_member___")


def test_vector_search_type_dense_sparse_distinct_same_labels_as_qdrant_vector() -> None:
    """Dense/sparse labels align between search mode and named vector slots."""
    assert VectorSearchType.DENSE.value == QdrantVectorType.DENSE.value
    assert VectorSearchType.SPARSE.value == QdrantVectorType.SPARSE.value
    assert VectorSearchType.HYBRID is not VectorSearchType.DENSE


def test_str_enum_json_roundtrip() -> None:
    payload = {
        "q": QuestionType.REASONING,
        "s": SectionType.DIAGNOSIS,
        "fusion": QdrantFusionMethod.DBSF,
        "mode": VectorSearchType.HYBRID,
    }
    assert json.dumps(payload) == (
        '{"q": "reasoning", "s": "diagnosis", "fusion": "dbsf", "mode": "hybrid"}'
    )
