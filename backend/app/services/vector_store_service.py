import hashlib
import logging
from typing import Protocol, List
import chromadb
from app.core.config import settings

from app.models.domain import EmbeddedDocument, VectorizedDocument, StorageSummary, RetrievedChunk

logger = logging.getLogger(__name__)

class VectorStore(Protocol):
    """
    Abstraction layer for vector storage. 
    Any orchestrating service must depend on this Protocol, never a concrete DB implementation.
    """
    def store(self, document: EmbeddedDocument) -> VectorizedDocument:
        ...

    def search(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        ...

    # Abstraction for background data sampling
    def get_background_embeddings(self, limit: int = 1000) -> List[List[float]]:
        ...


class ChromaVectorStore:
    """
    Concrete implementation of VectorStore using ChromaDB.
    """
    # Singleton instance to ensure persistence and prevent multi-load issues in FastAPI
    _client_instance = None

    def __init__(self, 
                collection_name: str = settings.VECTOR_COLLECTION,
                persist_directory: str = settings.VECTOR_DB_PATH):
        self.collection_name = collection_name
        self.provider = "chromadb"
        
        # Initialize client only once per application lifecycle
        if self.__class__._client_instance is None:
            logger.info(f"Initializing ChromaDB PersistentClient at: {persist_directory}")
            self.__class__._client_instance = chromadb.PersistentClient(path=persist_directory)
        
        self.client = self.__class__._client_instance
        
        # Ensure collection exists with strict ANN and distance metric settings
        logger.info(f"Accessing/Creating ChromaDB collection: '{self.collection_name}'")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",  # Enforce cosine similarity
                "hnsw:M": 16             # Default HNSW graph connections
            }
        )

    def store(self, document: EmbeddedDocument) -> VectorizedDocument:
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        doc_id = document.file_hash 
        
        # Clean, direct access to the filename!
        doc_filename = getattr(document.metadata, "filename", "unknown") or "unknown"

        logger.info(f"Preparing to store {len(document.chunks)} chunks for document_id: {doc_id} (filename: {doc_filename})")

        for i, chunk in enumerate(document.chunks):
            raw_id = f"{doc_id}_{i}_{chunk.text}"
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
            
            ids.append(chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            
            metadatas.append({
                "document_id": doc_id,
                "filename": doc_filename,               
                "page_number": getattr(chunk, "page_number", 1),
                "chunk_index": i,
                "model_name": getattr(chunk, "model_name", "unknown")
            })
            
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully upserted {len(ids)} chunks to collection '{self.collection_name}'.")
            
        except Exception as e:
            logger.error(f"Failed to upsert chunks to ChromaDB for document {doc_id}. Error: {str(e)}")
            raise

        return VectorizedDocument(
            document_id=doc_id,
            collection_name=self.collection_name,
            stored_chunks=len(ids),
            storage_summary=StorageSummary(
                provider=self.provider,
                collection_name=self.collection_name,
                total_chunks_stored=len(ids),
                persisted=True
            )
        )

    def search(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            chunks = []
            if not results.get("ids") or not results["ids"][0]:
                return chunks
                
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            # Safe extraction of nested query embeddings array from Chroma
            raw_embeddings = results.get("embeddings")
            embeddings = raw_embeddings[0] if (raw_embeddings and len(raw_embeddings) > 0) else []
            
            for i in range(len(ids)):
                score = 1.0 - distances[i]
                meta = metadatas[i] or {}

                # Convert numpy array to standard Python list if needed
                chunk_emb = embeddings[i] if i < len(embeddings) else None
                if hasattr(chunk_emb, "tolist"):
                    chunk_emb = chunk_emb.tolist()

                chunk = RetrievedChunk(
                    chunk_id=ids[i],
                    text=documents[i],
                    score=score,
                    document_id=meta.get("document_id", ""),
                    filename=meta.get("filename", ""),
                    page_number=meta.get("page_number"),
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta,
                    embedding=chunk_emb
                )
                chunks.append(chunk)
                
            return chunks
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def get_background_embeddings(self, limit: int = 1000) -> List[List[float]]:
        """
        Fetches all/sample vectors from ChromaDB to build the UMAP manifold.
        """
        try:
            total_count = self.collection.count()
            logger.info(f"ChromaDB Collection '{self.collection_name}' has {total_count} documents.")
            
            if total_count == 0:
                logger.warning("ChromaDB collection is empty! No background embeddings to fetch.")
                return []

            results = self.collection.get(
                limit=limit,
                include=["embeddings"]
            )
            
            embeddings = results.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                logger.warning("ChromaDB returned no embeddings in get_background_embeddings().")
                return []
            
            # Convert numpy ndarray to pure python list of lists
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
                
            return list(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to fetch background embeddings: {str(e)}")
            return []