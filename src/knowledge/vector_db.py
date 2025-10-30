"""
Vector database implementation for storing and retrieving embeddings.
Supports both FAISS and ChromaDB backends.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.config import config
from src.knowledge.zim_parser import DocumentChunk


# Type alias for clarity
EmbeddingVector = List[float]


class VectorDatabase:
    """
    Abstract base class for vector databases.
    """
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[EmbeddingVector],
    ) -> None:
        """Add documents with their embeddings to the database."""
        raise NotImplementedError
    
    def search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents.
        
        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        raise NotImplementedError
    
    def save(self, path: Path) -> None:
        """Save the database to disk."""
        raise NotImplementedError
    
    def load(self, path: Path) -> None:
        """Load the database from disk."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        raise NotImplementedError


class FAISSVectorDatabase(VectorDatabase):
    """
    FAISS-based vector database implementation.
    Fast and efficient for large-scale similarity search.
    """
    
    def __init__(self, dimension: Optional[int] = None):
        """
        Initialize FAISS vector database.
        
        Args:
            dimension: Dimension of embedding vectors
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")
        
        self.dimension = dimension or config.vector_db.vector_db_dimension
        self.index = self.faiss.IndexFlatL2(self.dimension)
        self.documents: List[DocumentChunk] = []
        
        logger.info(f"Initialized FAISS vector database with dimension {self.dimension}")
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[EmbeddingVector],
    ) -> None:
        """Add documents with their embeddings to the database."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-10)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store documents
        self.documents.extend(chunks)
        
        logger.debug(f"Added {len(chunks)} documents to FAISS database")
    
    def search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar documents."""
        if not self.documents:
            return []
        
        # Convert query to numpy array and normalize
        query_array = np.array([query_embedding], dtype=np.float32)
        query_norm = np.linalg.norm(query_array)
        query_array = query_array / (query_norm + 1e-10)
        
        # Search
        distances, indices = self.index.search(query_array, min(top_k, len(self.documents)))
        
        # Convert distances to similarity scores (1 - normalized_distance)
        # For L2 distance with normalized vectors, distance is in [0, 4]
        # We convert to similarity in [0, 1]
        similarities = 1 - (distances[0] / 4.0)
        
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(similarity)))
        
        return results
    
    def save(self, path: Path) -> None:
        """Save the database to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "faiss.index"
        self.faiss.write_index(self.index, str(index_path))
        
        # Save documents
        docs_path = path / "documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved FAISS database to {path}")
    
    def load(self, path: Path) -> None:
        """Load the database from disk."""
        index_path = path / "faiss.index"
        docs_path = path / "documents.pkl"
        
        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(f"Database files not found in {path}")
        
        # Load FAISS index
        self.index = self.faiss.read_index(str(index_path))
        
        # Load documents
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        logger.info(f"Loaded FAISS database from {path} ({len(self.documents)} documents)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "type": "FAISS",
            "dimension": self.dimension,
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
        }


class ChromaVectorDatabase(VectorDatabase):
    """
    ChromaDB-based vector database implementation.
    Easy to use with built-in metadata filtering.
    """
    
    def __init__(self, collection_name: str = "murmurnet_knowledge"):
        """
        Initialize ChromaDB vector database.
        
        Args:
            collection_name: Name of the collection
        """
        try:
            import chromadb
            from chromadb.config import Settings
            self.chromadb = chromadb
        except ImportError:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        # Initialize client
        db_path = config.vector_db.vector_db_path
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = self.chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "MurmurNet knowledge base"},
        )
        
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[EmbeddingVector],
    ) -> None:
        """Add documents with their embeddings to the database."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.debug(f"Added {len(chunks)} documents to ChromaDB")
    
    def search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar documents."""
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        
        # Convert results to DocumentChunk objects
        output = []
        
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                chunk = DocumentChunk(
                    chunk_id=results['ids'][0][i],
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                )
                similarity = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                output.append((chunk, similarity))
        
        return output
    
    def save(self, path: Path) -> None:
        """Save the database to disk."""
        # ChromaDB persists automatically
        logger.info("ChromaDB persists automatically")
    
    def load(self, path: Path) -> None:
        """Load the database from disk."""
        # ChromaDB loads automatically on initialization
        logger.info("ChromaDB loads automatically")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        count = self.collection.count()
        return {
            "type": "ChromaDB",
            "total_documents": count,
            "collection_name": self.collection.name,
        }


def create_vector_database(db_type: Optional[str] = None) -> VectorDatabase:
    """
    Factory function to create a vector database instance.
    
    Args:
        db_type: Type of database ('faiss' or 'chroma')
        
    Returns:
        VectorDatabase instance
    """
    db_type = db_type or config.vector_db.vector_db_type.value
    
    if db_type.lower() == "faiss":
        return FAISSVectorDatabase()
    elif db_type.lower() == "chroma":
        return ChromaVectorDatabase()
    else:
        raise ValueError(f"Unknown database type: {db_type}")
