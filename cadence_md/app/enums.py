"""
Shared string enums for clinical section labels, Qdrant wiring, and retrieval modes.
"""

from enum import StrEnum


class SectionType(StrEnum):
    """High-level clinical guideline section labels used when sampling generation contexts."""

    DEFINITION = "definition"
    SYMPTOMS = "symptoms"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    PREVENTION = "prevention"
    REHABILITATION = "rehabilitation"


class QdrantVectorType(StrEnum):
    """Named vector slots in the hybrid Qdrant collection (dense + sparse BM25)."""

    DENSE = "dense"
    SPARSE = "sparse"


class QdrantFusionMethod(StrEnum):
    """Prefetch fusion strategy when combining dense and sparse hits (hybrid search)."""

    RRF = "rrf"
    DBSF = "dbsf"


class VectorSearchType(StrEnum):
    """Which retrieval path ``QdrantManager.retrieve`` uses for a query."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
