from enum import StrEnum


class QuestionType(StrEnum):
    """Types of questions according to the RAGAS methodology"""

    SIMPLE = "simple"
    REASONING = "reasoning"
    MULTI_CONTEXT = "multi_context"
    CONDITIONAL = "conditional"
    COMPARISON = "comparison"


class SectionType(StrEnum):
    """Types of clinical recommendation sections"""

    DEFINITION = "definition"
    SYMPTOMS = "symptoms"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    PREVENTION = "prevention"
    REHABILITATION = "rehabilitation"


class RerankerAggregationStrategy(StrEnum):
    """Aggregation strategies for reranker"""

    L2_NORM = "l2_norm"
    MEAN = "mean"
    MAX = "max"


class QdrantVectorType(StrEnum):
    """Types of Qdrant vector types"""

    DENSE = "dense"
    SPARSE = "sparse"


class QdrantFusionMethod(StrEnum):
    """Types of Qdrant fusion methods"""

    RRF = "rrf"
    DBSF = "dbsf"


class VectorSearchType(StrEnum):
    """Types of vector search types"""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
