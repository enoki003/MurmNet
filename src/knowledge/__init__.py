"""Knowledge base and RAG system package."""

from src.knowledge.embeddings import EmbeddingModel
from src.knowledge.rag import RAGSystem, RetrievedDocument
from src.knowledge.vector_db import (
    VectorDatabase,
    FAISSVectorDatabase,
    ChromaVectorDatabase,
    create_vector_database,
)
from src.knowledge.zim_parser import ZIMParser, DocumentChunk

__all__ = [
    "EmbeddingModel",
    "RAGSystem",
    "RetrievedDocument",
    "VectorDatabase",
    "FAISSVectorDatabase",
    "ChromaVectorDatabase",
    "create_vector_database",
    "ZIMParser",
    "DocumentChunk",
]
