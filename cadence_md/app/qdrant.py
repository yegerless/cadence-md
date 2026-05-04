"""
Qdrant lifecycle, hybrid indexing, and retrieval for clinical guideline chunks.
"""

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.models import SparseVector
from tqdm import tqdm

from cadence_md.app.embedder import EmbedderWrapper
from cadence_md.app.enums import QdrantFusionMethod, QdrantVectorType, VectorSearchType
from cadence_md.app.pdf_parser import ClinicalGuidelinesParser, ClinicalSection
from cadence_md.app.settings import Settings, settings

# Configure logging for this module
logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Connect to Qdrant, (re)build a hybrid collection, index PDFs, and run retrieval.

    Top-level flow: :meth:`setup_qdrant` creates or validates the collection, optionally re-parses
    ``data_dir`` and uploads points. :meth:`retrieve` dispatches to dense, sparse, or hybrid search
    based on ``search_mode`` and ``fusion_method`` from settings.
    """

    def __init__(
        self,
        data_dir: Path,
        chunking_cfg: BaseModel,
        qdrant_cfg: BaseModel,
        url: str,
        api_key: str,
        https: bool,
        embedder: EmbedderWrapper,
        collection_name: str,
        uploading_batch_size: int,
        sparse_model: SparseTextEmbedding,
        search_mode: VectorSearchType,
        fusion_method: QdrantFusionMethod,
        sparse_top_k: int,
        dense_top_k: int,
        hybrid_top_k: int,
    ) -> None:
        """
        Store configuration, open a :class:`qdrant_client.QdrantClient`, and log readiness.

        Args:
            data_dir: Path to the directory containing the PDF files.
            chunking_cfg: Configuration for the chunking process.
            qdrant_cfg: Configuration for the Qdrant client.
            url: URL of the Qdrant server.
            api_key: API key for the Qdrant server.
            https: Whether to use HTTPS for the Qdrant server.
            embedder: Embedder wrapper.
            collection_name: Name of the collection to use.
            uploading_batch_size: Batch size for uploading points to Qdrant.
            sparse_model: Sparse model to use for the Qdrant client.
            search_mode: Search mode to use for the Qdrant client.
            fusion_method: Fusion method to use for the Qdrant client.
            sparse_top_k: Top k for sparse search.
            dense_top_k: Top k for dense search.
            hybrid_top_k: Top k for hybrid search.

        Raises:
            RuntimeError: If the Qdrant client cannot be constructed (bad URL, TLS, etc.).
        """
        # Use config if provided, otherwise use global settings
        self.data_dir = data_dir

        self.chunking_cfg = chunking_cfg
        self.qdrant_cfg = qdrant_cfg

        try:
            self.qdrant_client = QdrantClient(
                url=url,
                api_key=api_key,
                https=https,
            )
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Qdrant at {url}: {e}") from e

        # dense-embedder
        self.embedder = embedder

        self._collection_name = collection_name
        self.uploading_batch_size = uploading_batch_size

        # sparse BM25-encoder with FastEmbed
        self.sparse_model = sparse_model

        # retrieve config
        self.search_mode = search_mode
        self.fusion_method = fusion_method
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k
        self.hybrid_top_k = hybrid_top_k

        logger.info("QdrantManager initialized successfully")

    def _collection_exists(self) -> bool:
        """Return True if the configured collection name exists on the server."""
        try:
            cols = self.qdrant_client.get_collections().collections
        except Exception as e:
            raise RuntimeError(f"Cannot list Qdrant collections: {e}") from e
        return any(c.name == self._collection_name for c in cols)

    def _sparse_model_descriptor(self) -> str:
        """Human-readable sparse model id for collection metadata."""
        name = getattr(self.sparse_model, "model_name", None)
        if isinstance(name, str) and name:
            return name
        return type(self.sparse_model).__name__

    def _expected_collection_metadata(self) -> dict[str, str]:
        """Metadata describing embedding model and hybrid vector schema."""
        return {
            "embedding_model": str(self.embedder.model),
            "dense_vector_size": str(self.qdrant_cfg.vector_size),
            "distance": str(self.qdrant_cfg.distance),
            "sparse_modifier": str(qdrant_models.Modifier.IDF),
            "sparse_model": self._sparse_model_descriptor(),
        }

    def _write_collection_schema_metadata(self) -> None:
        """Persist schema descriptor on the collection (OpenAPI ``update_collection``)."""
        meta = self._expected_collection_metadata()
        try:
            self.qdrant_client.update_collection(
                collection_name=self._collection_name,
                metadata=meta,
            )
            logger.info("Updated Qdrant collection metadata for schema tracking")
        except Exception as e:
            msg = f"Failed to write collection metadata for '{self._collection_name}': {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def _read_collection_metadata(self) -> dict[str, str] | None:
        """
        Read collection metadata.

        Prefer ``config.metadata``; fallback ``info.metadata``.
        """
        info = self.qdrant_client.get_collection(collection_name=self._collection_name)
        config = getattr(info, "config", None)
        raw = getattr(config, "metadata", None) if config is not None else None
        if not raw:
            raw = getattr(info, "metadata", None)
        if not raw:
            return None
        return {str(k): str(v) for k, v in dict(raw).items()}

    def _collection_points_count(self) -> int:
        """Approximate/ exact point count for the collection."""
        try:
            cnt = self.qdrant_client.count(
                collection_name=self._collection_name,
                exact=True,
            )
            return int(cnt.count)
        except Exception as e:
            logger.warning(f"Could not count Qdrant points: {e}")
            return -1

    def ensure_collection_exists_and_schema_matches(self) -> None:
        """
        Validate collection exists, vector params, sparse config, and metadata.

        Raises:
            RuntimeError: On missing collection, schema mismatch, or incompatible metadata.
        """
        if not self._collection_exists():
            msg = (
                f"Qdrant collection '{self._collection_name}' does not exist. "
                "Set rebuild_collection=True for first-time indexing or create the collection."
            )
            raise RuntimeError(msg)

        info = self.qdrant_client.get_collection(collection_name=self._collection_name)
        params = info.config.params

        vectors = getattr(params, "vectors", None) or {}
        if QdrantVectorType.DENSE not in vectors:
            msg = (
                f"Collection '{self._collection_name}' missing dense vector "
                f"'{QdrantVectorType.DENSE}'"
            )
            raise RuntimeError(msg)
        dense = vectors[QdrantVectorType.DENSE]
        if int(dense.size) != int(self.qdrant_cfg.vector_size):
            msg = (
                f"Dense vector size mismatch: expected {self.qdrant_cfg.vector_size}, "
                f"got {dense.size}. Re-index with rebuild_collection=True."
            )
            raise RuntimeError(msg)
        if str(dense.distance) != str(self.qdrant_cfg.distance):
            msg = (
                f"Dense vector distance mismatch: expected {self.qdrant_cfg.distance}, "
                f"got {dense.distance}. Re-index with rebuild_collection=True."
            )
            raise RuntimeError(msg)

        sparse_vectors = getattr(params, "sparse_vectors", None) or {}
        if QdrantVectorType.SPARSE not in sparse_vectors:
            msg = (
                f"Collection '{self._collection_name}' missing sparse vector "
                f"'{QdrantVectorType.SPARSE}'"
            )
            raise RuntimeError(msg)
        sparse = sparse_vectors[QdrantVectorType.SPARSE]
        if str(sparse.modifier) != str(qdrant_models.Modifier.IDF):
            msg = (
                f"Sparse vector modifier mismatch: expected {qdrant_models.Modifier.IDF}, "
                f"got {sparse.modifier}. Re-index with rebuild_collection=True."
            )
            raise RuntimeError(msg)

        expected_meta = self._expected_collection_metadata()
        meta = self._read_collection_metadata()
        if not meta:
            n_pts = self._collection_points_count()
            if n_pts == 0:
                logger.warning(
                    "Collection metadata missing on an empty collection; writing expected metadata"
                )
                self._write_collection_schema_metadata()
                meta = self._read_collection_metadata()
            else:
                msg = (
                    f"Collection '{self._collection_name}' has points but no cadence metadata. "
                    "Re-index with rebuild_collection=True to attach schema metadata."
                )
                raise RuntimeError(msg)

        if not meta:
            msg = (
                f"Collection '{self._collection_name}' metadata is still missing "
                "after repair attempt"
            )
            raise RuntimeError(msg)

        for key, expected_val in expected_meta.items():
            if meta.get(key) != expected_val:
                msg = (
                    f"Collection metadata mismatch on {key!r}: expected {expected_val!r}, "
                    f"got {meta.get(key)!r}. Re-index with rebuild_collection=True after aligning "
                    "embedding model / vector_size / sparse settings."
                )
                raise RuntimeError(msg)

    def _create_collection(self) -> None:
        """Create the hybrid collection or delete/recreate it when ``rebuild_collection`` is set.

        On success for a new collection, writes schema metadata for later validation in
        :meth:`ensure_collection_exists_and_schema_matches`.
        """
        # try connect to Qdrant
        try:
            collections = self.qdrant_client.get_collections()
            logger.info(f"Qdrant is reachable. Collections: {len(collections.collections)}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Qdrant: {e}") from e

        # Check if collection self.qdrant_cfg.collection_name exists
        try:
            existing = self.qdrant_client.get_collection(self._collection_name)
            collection_exists = existing is not None
        except Exception as e:
            logger.warning(f"Error checking collection existence: {e}")
            collection_exists = False

        if collection_exists and self.qdrant_cfg.rebuild_collection:
            try:
                self.qdrant_client.delete_collection(self._collection_name)
                logger.info(f"Deleted existing collection '{self._collection_name}'")
            except Exception as e:
                logger.error(f"Failed to delete collection '{self._collection_name}': {e}")
                raise
            collection_exists = False

        if not collection_exists:
            try:
                # Collection with named dense + sparse vectors
                self.qdrant_client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config={
                        QdrantVectorType.DENSE: qdrant_models.VectorParams(
                            size=self.qdrant_cfg.vector_size,
                            distance=self.qdrant_cfg.distance,
                            on_disk=True,
                        ),
                    },
                    sparse_vectors_config={
                        QdrantVectorType.SPARSE: qdrant_models.SparseVectorParams(
                            modifier=qdrant_models.Modifier.IDF,
                        ),
                    },
                )
                logger.info(
                    f"Created hybrid collection '{self._collection_name}' "
                    "(dense + sparse: Modifier.IDF)"
                )
                self._write_collection_schema_metadata()
            except Exception as e:
                logger.error(f"Failed to create collection '{self._collection_name}': {e}")
                raise
        else:
            logger.info(f"Collection '{self._collection_name}' already exists")

    def __parse_pdf_dir(self, pdf_dir: Path) -> list[ClinicalSection]:
        """Parse all parseable PDFs under ``pdf_dir`` into :class:`ClinicalSection` records."""
        logger.info(f"Init ClinicalGuidelinesParser for directory: {pdf_dir}")

        parser = ClinicalGuidelinesParser()

        try:
            sections = parser.parse_directory(pdf_dir)
        except Exception as e:
            logger.error(f"Error parsing PDF directory '{pdf_dir}': {e}")
            raise RuntimeError(f"Failed to parse PDF files in '{pdf_dir}': {e}") from e

        if not sections:
            logger.warning(f"No clinical sections found in directory: {pdf_dir}")

        logger.info(f"Obtained {len(sections)} clinical sections from files in {pdf_dir}")
        return sections

    @staticmethod
    def _metadata_group_key(metadata: dict[str, object]) -> str:
        """Stable key to group chunks from the same clinical section.

        Prefer ``section_id``; otherwise hash ``filename`` + ``section_title``.
        """
        sid = metadata.get("section_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
        fn = str(metadata.get("filename", "unknown"))
        st = str(metadata.get("section_title", ""))
        raw = f"{fn}|{st}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _chunk_id_prefix(self, metadata: dict[str, object], group_key: str) -> str:
        """Prefix for ``chunk_id``: ``section_id`` when present, else ``fallback_<group_hash>``."""
        sid = metadata.get("section_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
        return f"fallback_{group_key}"

    def _attach_chunk_metadata(self, chunks: list[Document]) -> list[Document]:
        """Set chunk_index, chunk_total, chunk_id, chunk_size, chunk_overlap on each chunk."""
        if not chunks:
            return chunks

        totals: dict[str, int] = defaultdict(int)
        for doc in chunks:
            totals[self._metadata_group_key(doc.metadata or {})] += 1

        seen: dict[str, int] = defaultdict(int)
        chunk_size = int(getattr(self.chunking_cfg, "chunk_size", 0))
        chunk_overlap = int(getattr(self.chunking_cfg, "chunk_overlap", 0))

        for doc in chunks:
            meta = dict(doc.metadata or {})
            gk = self._metadata_group_key(meta)
            idx = seen[gk]
            seen[gk] += 1
            prefix = self._chunk_id_prefix(meta, gk)
            meta["chunk_index"] = idx
            meta["chunk_total"] = totals[gk]
            meta["chunk_id"] = f"{prefix}_chunk_{idx:04d}"
            meta["chunk_size"] = chunk_size
            meta["chunk_overlap"] = chunk_overlap
            doc.metadata = meta

        return chunks

    def __create_chunks(self, sections: list[ClinicalSection]) -> list[Document]:
        """Turn sections into LangChain documents, split them, then enrich chunk metadata.

        After splitting, :meth:`_attach_chunk_metadata` adds ``chunk_id``, ordinals, and splitter
        hyperparameters for traceability.
        """
        logger.info("Creating text chunks from clinical sections")

        documents: list[Document] = []
        for section in sections:
            try:
                section_dict = section.to_dict()
                content = section_dict.pop("content")
                doc = Document(page_content=content, metadata=section_dict)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error creating document from section: {e}")
                continue

        if not documents:
            raise ValueError("No valid documents created from sections")

        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunking_cfg.chunk_size,
                chunk_overlap=self.chunking_cfg.chunk_overlap,
                separators=self.chunking_cfg.separators,
            )
            chunks = splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise

        chunks = self._attach_chunk_metadata(chunks)

        logger.info(f"Split {len(sections)} sections into {len(chunks)} chunks")
        return chunks

    def _index_chunks(self, pdf_dir: Path) -> None:
        """End-to-end path from PDFs to uploaded Qdrant points (dense + sparse in parallel)."""
        logger.info("Indexing chunks into Qdrant (dense + BM25 sparse)...")

        sections = self.__parse_pdf_dir(pdf_dir)
        chunks = self.__create_chunks(sections)

        if not chunks:
            logger.warning("No chunks to index")
            return

        texts = [chunk.page_content for chunk in chunks]
        logger.info(f"Indexing {len(chunks)} chunks into Qdrant (dense + BM25 sparse)...")

        batch_size = self.uploading_batch_size
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Indexing batches", total=num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            # dense embeddings
            try:
                dense_embeddings = self.embedder.encode(batch_texts)
            except TypeError as e:
                logger.error(f"Error encoding dense embeddings for batch {batch_idx}: {e}")
                continue

            # sparse BM25 via FastEmbed
            # embed(...) returns a generator of SparseEmbedding objects
            try:
                sparse_embeddings = list(self.sparse_model.embed(batch_texts))
            except Exception as e:
                logger.error(f"Error encoding sparse embeddings for batch {batch_idx}: {e}")
                continue

            if len(dense_embeddings) != len(sparse_embeddings):
                logger.error(
                    f"Mismatch in embedding sizes for batch {batch_idx}: "
                    f"dense={len(dense_embeddings)}, sparse={len(sparse_embeddings)}"
                )
                continue

            points: list[qdrant_models.PointStruct] = []
            for i, (dense_vec, sparse_emb) in enumerate(zip(dense_embeddings, sparse_embeddings)):  # noqa: B905
                global_idx = start + i

                try:
                    sparse_vec = SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    )
                except Exception as e:
                    logger.error(f"Error creating sparse vector for chunk {global_idx}: {e}")
                    continue

                try:
                    point = qdrant_models.PointStruct(
                        id=global_idx,
                        vector={
                            QdrantVectorType.DENSE: dense_vec,
                            QdrantVectorType.SPARSE: sparse_vec,
                        },
                        payload={
                            "text": texts[global_idx],
                            **chunks[global_idx].metadata,
                        },
                    )
                    points.append(point)
                except Exception as e:
                    logger.error(f"Error creating point for chunk {global_idx}: {e}")
                    continue

            if not points:
                logger.warning(f"No valid points in batch {batch_idx}, skipping")
                continue

            try:
                self.qdrant_client.upload_points(
                    collection_name=self._collection_name,
                    points=points,
                    batch_size=self.uploading_batch_size,
                    wait=True,
                )
            except Exception as e:
                logger.error(f"Failed to upload points for batch {batch_idx}: {e}")
                raise

        logger.info("Indexed all chunks into Qdrant (dense + sparse BM25)")

    def setup_qdrant(self) -> None:
        """Entry point: rebuild or create collection, then validate schema/metadata.

        - ``rebuild_collection=True``: delete/recreate (if needed), full re-index from ``data_dir``.
        - Missing collection without rebuild: create empty hybrid collection for later indexing.
        - Otherwise: validate existing collection matches embedder and sparse configuration.
        """
        logger.info("Setting up Qdrant database for RAG operations")

        if self.qdrant_cfg.rebuild_collection:
            logger.info(f"Rebuilding collection '{self._collection_name}'")
            self._create_collection()
            self._index_chunks(self.data_dir)
        elif not self._collection_exists():
            logger.info(
                "Collection '%s' is missing; creating an empty hybrid collection",
                self._collection_name,
            )
            self._create_collection()
        else:
            logger.info(
                "Using existing collection '%s'; validating schema/metadata",
                self._collection_name,
            )

        self.ensure_collection_exists_and_schema_matches()

        logger.info("Qdrant database is ready for working with RAG")

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve top passages as LangChain documents with Qdrant scores (mode from settings).

        Returns:
            List of ``(Document, score)`` pairs ordered by the backend (fusion for hybrid).
        """
        if self.search_mode == VectorSearchType.DENSE:
            return self._retrieve_dense(query)
        if self.search_mode == VectorSearchType.SPARSE:
            return self._retrieve_sparse(query)
        if self.search_mode == VectorSearchType.HYBRID:
            return self._retrieve_hybrid(query)
        raise ValueError(f"Unknown search_mode: {self.search_mode}")

    def _retrieve_dense(self, query: str) -> list[tuple[Document, float]]:
        """Dense-only retrieval: ``encode_query`` → named vector ``dense``."""
        logger.debug(f"Performing dense retrieval for query: {query[:50]}...")

        try:
            query_vec = self.embedder.encode_query(query)
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise RuntimeError(f"Failed to encode query '{query}': {e}") from e

        try:
            res = self.qdrant_client.query_points(
                collection_name=self._collection_name,
                query=query_vec,
                using=QdrantVectorType.DENSE,
                with_payload=True,
                limit=self.dense_top_k,
            )
        except Exception as e:
            logger.error(f"Error querying Qdrant for query '{query}': {e}")
            raise RuntimeError(f"Failed to query Qdrant: {e}") from e

        return self._scored_points_to_document_score_pairs(res.points)

    def _retrieve_sparse(self, query: str) -> list[tuple[Document, float]]:
        """Sparse-only retrieval: FastEmbed query embedding → named vector ``sparse``."""
        logger.debug(f"Performing sparse retrieval for query: {query[:50]}...")

        try:
            # embed() returns a generator, convert to list
            sparse_embs = list(self.sparse_model.embed([query]))

            if not sparse_embs:
                raise ValueError("No sparse embedding generated for query")

            sparse_emb = sparse_embs[0]
            sparse_vec = SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            )
        except Exception as e:
            logger.error(f"Error encoding sparse embedding for query '{query}': {e}")
            raise RuntimeError(f"Failed to encode sparse embedding: {e}") from e

        try:
            res = self.qdrant_client.query_points(
                collection_name=self._collection_name,
                query=sparse_vec,
                using=QdrantVectorType.SPARSE,
                with_payload=True,
                limit=self.sparse_top_k,
            )
        except Exception as e:
            logger.error(f"Error querying Qdrant sparse collection: {e}")
            raise RuntimeError(f"Failed to query Qdrant sparse collection: {e}") from e

        return self._scored_points_to_document_score_pairs(res.points)

    def _retrieve_hybrid(self, query: str) -> list[tuple[Document, float]]:
        """Prefetch sparse and dense top-k lists, then fuse (RRF or DBSF) to a final ranking."""
        logger.debug(f"Performing hybrid retrieval for query: {query[:50]}...")

        # dense query
        try:
            dense_vec = self.embedder.encode_query(query)
        except Exception as e:
            logger.error(f"Error encoding dense query: {e}")
            raise RuntimeError(f"Failed to encode dense embedding: {e}") from e

        # sparse query via FastEmbed
        try:
            sparse_embs = list(self.sparse_model.embed([query]))

            if not sparse_embs:
                raise ValueError("No sparse embedding generated for query")

            sparse_emb = sparse_embs[0]
            sparse_vec = SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            )
        except Exception as e:
            logger.error(f"Error encoding sparse embedding: {e}")
            raise RuntimeError(f"Failed to encode sparse embedding: {e}") from e

        # Configure fusion method
        fusion_method = self.fusion_method.lower()
        if fusion_method == QdrantFusionMethod.RRF:
            fusion_query = qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF)
        elif fusion_method == QdrantFusionMethod.DBSF:
            fusion_query = qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.DBSF)
        else:
            logger.error(f"Unknown fusion_method: {self.fusion_method}")
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        try:
            res = self.qdrant_client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    qdrant_models.Prefetch(
                        query=sparse_vec,
                        using=QdrantVectorType.SPARSE,
                        limit=self.sparse_top_k,
                    ),
                    qdrant_models.Prefetch(
                        query=dense_vec,
                        using=QdrantVectorType.DENSE,
                        limit=self.dense_top_k,
                    ),
                ],
                query=fusion_query,
                with_payload=True,
                limit=self.hybrid_top_k,
            )
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise RuntimeError(f"Failed to perform hybrid search: {e}") from e

        return self._scored_points_to_document_score_pairs(res.points)

    @staticmethod
    def _scored_points_to_document_score_pairs(
        points: list[qdrant_models.ScoredPoint],
    ) -> list[tuple[Document, float]]:
        """Map Qdrant scored points to ``Document``; payload ``text`` becomes ``page_content``."""
        pairs: list[tuple[Document, float]] = []
        for pt in points:
            payload = dict(pt.payload or {})
            page_content = payload.pop("text", "")
            score = float(pt.score) if pt.score is not None else float("nan")
            doc = Document(page_content=page_content, metadata=payload)
            pairs.append((doc, score))
        return pairs


def get_qdrant_manager_from_settings(
    embedder: EmbedderWrapper,
    app_settings: Settings = settings,
) -> QdrantManager:
    """Construct :class:`QdrantManager` from ``Settings`` (chunking, Qdrant, retrieval sub-configs).

    The embedder is injected so indexing and query encoding always share the same dense model
    instance and instruction settings as the rest of the app.
    """
    qc = app_settings.rag_config.qdrant_config
    retr = app_settings.rag_config.retrieval
    return QdrantManager(
        data_dir=qc.data_dir,
        chunking_cfg=app_settings.rag_config.chunking,
        qdrant_cfg=qc,
        url=app_settings.QDRANT_BASE_URL,
        api_key=app_settings.QDRANT_API_KEY,
        https=app_settings.QDRANT_HTTPS,
        embedder=embedder,
        collection_name=qc.collection_name,
        uploading_batch_size=qc.uploading_batch_size,
        sparse_model=qc.sparse_model,
        search_mode=retr.search_mode,
        fusion_method=retr.fusion_method,
        sparse_top_k=retr.sparse_top_k,
        dense_top_k=retr.dense_top_k,
        hybrid_top_k=retr.hybrid_top_k,
    )
