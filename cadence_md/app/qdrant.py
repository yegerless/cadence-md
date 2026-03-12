import logging
from pathlib import Path

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.models import SparseVector
from tqdm import tqdm

from cadence_md.app.embedder import EmbedderWrapper
from cadence_md.app.enums import QdrantFusionMethod, QdrantVectorType, VectorSearchType
from cadence_md.app.pdf_parser import ClinicalGuidelinesParser, ClinicalSection
from cadence_md.app.settings import RAGConfig

# Configure logging for this module
logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Manages Qdrant vector database operations for RAG (Retrieval-Augmented Generation) system.

    This class provides functionality to:
    - Create and manage Qdrant collections with dense and sparse vector support
    - Index PDF documents into the vector database
    - Retrieve documents using dense, sparse, or hybrid search modes

    Attributes:
        chunking_cfg: Configuration for text chunking strategy
        retrieval_cfg: Configuration for document retrieval settings
        qdrant_cfg: Qdrant database configuration
        qdrant_client: Qdrant client instance for database operations
        embedder: Embedding wrapper for generating vector embeddings
        sparse_model: FastEmbed BM25 sparse embedding model
    """

    def __init__(self, config: RAGConfig, embedder: EmbedderWrapper) -> None:
        """
        Initialize the QdrantManager with configuration and embedder.

        Args:
            config: RAGConfig instance containing all configuration settings
            embedder: EmbedderWrapper instance for generating embeddings

        Raises:
            RuntimeError: If connection to Qdrant database cannot be established
        """
        self.chunking_cfg = config.chunking
        self.retrieval_cfg = config.retrieval
        self.qdrant_cfg = config.qdrant_config

        try:
            self.qdrant_client = QdrantClient(
                host=self.qdrant_cfg.host,
                port=self.qdrant_cfg.port,
                api_key=self.qdrant_cfg.api_key,
                https=self.qdrant_cfg.https,
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Qdrant at {self.qdrant_cfg.host}:{self.qdrant_cfg.port}: {e}"
            ) from e

        # dense-embedder
        self.embedder = embedder
        self._collection_name = self.qdrant_cfg.collection_name
        self.uploading_batch_size = self.qdrant_cfg.uploading_batch_size

        # sparse BM25-encoder with FastEmbed
        self.sparse_model: SparseTextEmbedding = SparseTextEmbedding(model_name="Qdrant/bm25")

        # retrieve config
        self.search_mode: str = self.retrieval_cfg.search_mode
        self.fusion_method: str = self.retrieval_cfg.fusion_method
        self.sparse_top_k: int = self.retrieval_cfg.sparse_top_k
        self.dense_top_k: int = self.retrieval_cfg.dense_top_k
        self.hybrid_top_k: int = self.retrieval_cfg.hybrid_top_k

        logger.info("QdrantManager initialized successfully")

    def _create_collection(self) -> None:
        """
        Create a new Qdrant collection with dense and sparse vector support.

        Checks if the target collection already exists. If it exists and
        rebuild_collection is enabled, deletes the existing collection first.

        Creates a hybrid collection with:
        - Dense vectors using configured size and distance metric
        - Sparse vectors using BM25 with IDF modifier

        Raises:
            RuntimeError: If Qdrant database is not reachable or collection creation fails
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
            except Exception as e:
                logger.error(f"Failed to create collection '{self._collection_name}': {e}")
                raise
        else:
            logger.info(f"Collection '{self._collection_name}' already exists")

    def __parse_pdf_dir(self, pdf_dir: Path) -> list[ClinicalSection]:
        """
        Parse all PDF files in the specified directory.

        Args:
            pdf_dir: Path to directory containing PDF files

        Returns:
            List of ClinicalSection objects extracted from PDFs

        Raises:
            RuntimeError: If parsing fails or no files are found
        """
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

    def __create_chunks(self, sections: list[ClinicalSection]) -> list[Document]:
        """
        Convert ClinicalSection objects into Document chunks for indexing.

        Args:
            sections: List of ClinicalSection objects to chunk

        Returns:
            List of Document objects with text chunks

        Raises:
            ValueError: If sections list is empty or document creation fails
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
            )
            chunks = splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise

        logger.info(f"Split {len(sections)} sections into {len(chunks)} chunks")
        return chunks

    def _index_chunks(self, pdf_dir: Path) -> None:
        """
        Index all text chunks from PDF files into Qdrant database.

        Processes documents in batches and indexes both dense and sparse
        vectors for hybrid search capabilities. Uses tqdm progress bar
        to track indexing progress.

        Args:
            pdf_dir: Path to directory containing PDF files to index

        Raises:
            RuntimeError: If indexing fails for any batch
        """
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

    def setup_qdrant(self, data_dir: Path) -> None:
        """
        Set up Qdrant collection for RAG operations.

        If rebuild_collection is enabled, deletes existing collection
        and re-indexes all documents. Otherwise, uses the existing
        collection without modifications.

        Args:
            data_dir: Path to directory containing PDF files for indexing
        """
        logger.info("Setting up Qdrant database for RAG operations")

        if self.qdrant_cfg.rebuild_collection:
            logger.info(f"Rebuilding collection '{self._collection_name}'")
            self._create_collection()
            self._index_chunks(data_dir)
        else:
            logger.info(
                f"Using existing collection '{self._collection_name}'; FastEmbed BM25 is stateless"
            )

        logger.info("Qdrant database is ready for working with RAG")

    def retrieve_documents(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents using the configured search mode.

        Args:
            query: Search query text

        Returns:
            List of relevant Document objects sorted by relevance

        Raises:
            ValueError: If search_mode is not one of 'dense', 'sparse', or 'hybrid'
        """
        if self.search_mode == VectorSearchType.DENSE:
            return self._retrieve_dense(query)
        if self.search_mode == VectorSearchType.SPARSE:
            return self._retrieve_sparse(query)
        if self.search_mode == VectorSearchType.HYBRID:
            return self._retrieve_hybrid(query)
        raise ValueError(f"Unknown search_mode: {self.search_mode}")

    def _retrieve_dense(self, query: str) -> list[Document]:
        """
        Retrieve documents using dense vector similarity search.

        Converts the query to a dense embedding and searches the
        Qdrant collection for similar documents.

        Args:
            query: Text query to search for

        Returns:
            List of relevant Document objects

        Raises:
            RuntimeError: If query embedding or search fails
        """
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
                limit=self.sparse_top_k,
            )
        except Exception as e:
            logger.error(f"Error querying Qdrant for query '{query}': {e}")
            raise RuntimeError(f"Failed to query Qdrant: {e}") from e

        return self._scored_points_to_documents(res.points)

    def _retrieve_sparse(self, query: str) -> list[Document]:
        """
        Retrieve documents using sparse BM25 vector search.

        Converts the query to a sparse embedding using FastEmbed's
        BM25 model and searches the Qdrant collection.

        Args:
            query: Text query to search for

        Returns:
            List of relevant Document objects

        Raises:
            RuntimeError: If query embedding or search fails
        """
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

        return self._scored_points_to_documents(res.points)

    def _retrieve_hybrid(self, query: str) -> list[Document]:
        """
        Retrieve documents using hybrid dense + sparse search with fusion.

        Combines results from both dense and sparse vector searches
        using the configured fusion method (RRF or DBSF). This provides
        a balanced retrieval strategy leveraging both semantic and
        lexical matching capabilities.

        Args:
            query: Text query to search for

        Returns:
            List of relevant Document objects

        Raises:
            ValueError: If fusion_method is not 'rrf' or 'dbsf'
            RuntimeError: If embedding or search fails
        """
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

        return self._scored_points_to_documents(res.points)

    @staticmethod
    def _scored_points_to_documents(
        points: list[qdrant_models.ScoredPoint],
    ) -> list[Document]:
        """
        Convert Qdrant scored points to LangChain Document objects.

        Args:
            points: List of ScoredPoint objects from Qdrant

        Returns:
            List of Document objects with metadata and scores

        Note:
            The 'text' field is removed from payload and used as
            page_content. The original score is added to metadata.
        """
        docs: list[Document] = []
        for pt in points:
            payload = pt.payload or {}
            page_content = payload.pop("text", "")

            # Add score to metadata
            if pt.score is not None:
                payload["score"] = pt.score

            doc = Document(page_content=page_content, metadata=payload)
            docs.append(doc)
        return docs
