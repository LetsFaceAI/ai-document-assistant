import hashlib
import logging
from typing import Protocol
import chromadb

from app.models.domain import EmbeddedDocument, VectorizedDocument, StorageSummary

logger = logging.getLogger(__name__)

class VectorStore(Protocol):
    """
    Abstraction layer for vector storage. 
    Any orchestrating service must depend on this Protocol, never a concrete DB implementation.
    """
    def store(self, document: EmbeddedDocument) -> VectorizedDocument:
        ...


class ChromaVectorStore:
    """
    Concrete implementation of VectorStore using ChromaDB.
    """
    # Singleton instance to ensure persistence and prevent multi-load issues in FastAPI
    _client_instance = None

    def __init__(self, persist_directory: str, collection_name: str):
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
        
        logger.info(f"Preparing to store {len(document.chunks)} chunks for document_id: {doc_id}")

        for i, chunk in enumerate(document.chunks):
            # Generate deterministic, stable IDs using content hashing
            # This prevents duplicates if the same document is re-embedded and re-inserted
            raw_id = f"{doc_id}_{i}_{chunk.text}"
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
            
            ids.append(chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            
            # Pack metadata required for filtering at retrieval time
            # Raw full text is stored externally; we only store chunk text and pointers here
            metadatas.append({
                "document_id": doc_id,
                "filename": getattr(document.metadata, "file_name", "unknown"),                
                "page_number": getattr(chunk, 'page_number', 1), # Fallback to 1 if missing
                "chunk_index": i,
                "model_name": chunk.model_name
            })
            
        try:
            # Use upsert (idempotent) instead of add to gracefully handle re-indexing operations
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

        # Map back to the Domain Model to prevent leaking ChromaDB objects to the caller
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