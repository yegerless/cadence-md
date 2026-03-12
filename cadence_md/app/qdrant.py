import math
from pathlib import Path

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.models import SparseVector
from tqdm import tqdm

from cadence_md.app.embedder import EmbedderWrapper
from cadence_md.app.enums import QdrantVectorType
from cadence_md.app.pdf_parser import ClinicalGuidelinesParser, ClinicalSection
from cadence_md.app.settings import RAGConfig


class QdrantManager:
    """ """

    def __init__(self, config: RAGConfig, embedder: EmbedderWrapper) -> None:
        """ """
        self.chunking_cfg = config.chunking
        self.retrieval_cfg = config.retrieval
        self.qdrant_cfg = config.qdrant_config

        self.qdrant_client = QdrantClient(
            host=self.qdrant_cfg.host,
            port=self.qdrant_cfg.port,
            api_key=self.qdrant_cfg.api_key,
            https=self.qdrant_cfg.https,
        )

        # dense-embedder
        self.embedder = embedder
        self.max_length = config.embedding.max_length  # TODO: not used
        self.uploading_batch_size = self.qdrant_cfg.uploading_batch_size

        # sparse BM25-encoder with FastEmbed
        self.sparse_model: SparseTextEmbedding = SparseTextEmbedding(model_name="Qdrant/bm25")

        # retrieve config
        self.search_mode: str = self.retrieval_cfg.search_mode
        self.fusion_method: str = self.retrieval_cfg.fusion_method
        self.sparse_top_k: int = self.retrieval_cfg.sparse_top_k
        self.dense_top_k: int = self.retrieval_cfg.dense_top_k
        self.hybrid_top_k: int = self.retrieval_cfg.hybrid_top_k

    def _create_collection(self) -> None:
        try:
            collections = self.qdrant_client.get_collections()
            print(f"Qdrant is reachable. Collections: {len(collections.collections)}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Qdrant: {e}") from e

        collection_name = self.qdrant_cfg.collection_name

        try:
            existing = self.qdrant_client.get_collection(collection_name)
            collection_exists = existing is not None
        except Exception:
            collection_exists = False

        if collection_exists and self.qdrant_cfg.rebuild_collection:
            self.qdrant_client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'")
            collection_exists = False

        if not collection_exists:
            # Collection with named dense + sparse vectors
            self.qdrant_client.create_collection(
                collection_name=collection_name,
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
            print(f"✓ Created hybrid collection '{collection_name}' (dense + sparse: Modifier.IDF)")
        else:
            print(f"Collection '{collection_name}' already exists")

    def __parse_pdf_dir(self, pdf_dir: Path) -> list[ClinicalSection]:
        """ """
        parser = ClinicalGuidelinesParser()
        print("Init ClinicalGuidelinesParser")
        sections = parser.parse_directory(pdf_dir)
        print(f"Obtained {len(sections)} from files in {pdf_dir}")
        return sections

    def __create_chunks(self, sections: list[ClinicalSection]) -> list[Document]:
        """ """
        documents: list[Document] = []
        for section in sections:
            section_dict = section.to_dict()
            content = section_dict.pop("content")
            doc = Document(page_content=content, metadata=section_dict)
            documents.append(doc)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunking_cfg.chunk_size,
            chunk_overlap=self.chunking_cfg.chunk_overlap,
        )
        print("Init splitter")
        chunks = splitter.split_documents(documents)
        print(f"Split sections on {len(chunks)} chunks")
        return chunks

    def _index_chunks(self, pdf_dir: Path) -> None:
        """ """
        sections = self.__parse_pdf_dir(pdf_dir)
        chunks = self.__create_chunks(sections)
        texts = [chunk.page_content for chunk in chunks]

        print(f"Indexing {len(chunks)} chunks into Qdrant (dense + BM25 sparse)...")

        batch_size = self.uploading_batch_size
        num_batches = math.ceil(len(texts) / batch_size)

        for batch_idx in tqdm(range(num_batches), desc="Indexing batches"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            # dense
            try:
                dense_embeddings = self.embedder.encode(batch_texts)
            except TypeError as e:
                print(f"Error encoding dense embeddings: {e}")
                continue

            # sparse BM25 via FastEmbed
            # embed(...) возвращает генератор SparseEmbedding
            sparse_embeddings = list(self.sparse_model.embed(batch_texts))

            points: list[qdrant_models.PointStruct] = []
            for i, (dense_vec, sparse_emb) in enumerate(
                zip(dense_embeddings, sparse_embeddings, strict=True)
            ):
                global_idx = start + i

                sparse_vec = SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist(),
                )

                points.append(
                    qdrant_models.PointStruct(
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
                )

            self.qdrant_client.upload_points(
                collection_name=self.qdrant_cfg.collection_name,
                points=points,
                batch_size=self.uploading_batch_size,
                wait=True,
            )

        print("✓ Indexed all chunks into Qdrant (dense + sparse BM25)")

    def setup_qdrant(self, data_dir: Path) -> None:
        """ """
        if self.qdrant_cfg.rebuild_collection:
            self._create_collection()
            self._index_chunks(data_dir)
        else:
            print("Using existing collection; FastEmbed BM25 is stateless")
        print("Qdrant database is ready for working with RAG")

    def retrieve_documents(self, query: str) -> list[Document]:
        """ """
        if self.search_mode == "dense":
            return self._retrieve_dense(query)
        if self.search_mode == "sparse":
            return self._retrieve_sparse(query)
        if self.search_mode == "hybrid":
            return self._retrieve_hybrid(query)
        raise ValueError(f"Unknown search_mode: {self.search_mode}")

    def _retrieve_dense(self, query: str) -> list[Document]:
        """ """
        query_vec = self.embedder.encode_query(query)
        res = self.qdrant_client.query_points(
            collection_name=self.qdrant_cfg.collection_name,
            query=query_vec,
            using=QdrantVectorType.DENSE,
            with_payload=True,
            limit=self.retrieval_cfg.top_k,
        )
        return self._scored_points_to_documents(res.points)

    def _retrieve_sparse(self, query: str) -> list[Document]:
        """ """
        sparse_emb = next(self.sparse_model.embed([query]))
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        res = self.qdrant_client.query_points(
            collection_name=self.qdrant_cfg.collection_name,
            query=sparse_vec,
            using=QdrantVectorType.SPARSE,
            with_payload=True,
            limit=self.retrieval_cfg.top_k,
        )
        return self._scored_points_to_documents(res.points)

    def _retrieve_hybrid(self, query: str) -> list[Document]:
        # dense query
        dense_vec = self.embedder.encode_query(query)

        # sparse query via FastEmbed
        sparse_emb = next(self.sparse_model.embed([query]))
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        # Choice fusion method
        if self.fusion_method.lower() == "rrf":
            fusion_query = qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF)
        elif self.fusion_method.lower() == "dbsf":
            fusion_query = qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.DBSF)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        res = self.qdrant_client.query_points(
            collection_name=self.qdrant_cfg.collection_name,
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
        return self._scored_points_to_documents(res.points)

    @staticmethod
    def _scored_points_to_documents(
        points: list[qdrant_models.ScoredPoint],
    ) -> list[Document]:
        docs: list[Document] = []
        for pt in points:
            payload = pt.payload
            page_content = payload.pop("text", "")
            payload["score"] = pt.score
            docs.append(Document(page_content=page_content, metadata=payload))
        return docs
