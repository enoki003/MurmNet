"""
RAG (Retrieval-Augmented Generation) system.
Combines knowledge retrieval with generation capabilities.
"""

from typing import Dict, List, Optional

from loguru import logger

from src.config import config
from src.knowledge.embeddings import EmbeddingModel
from src.knowledge.vector_db import VectorDatabase, create_vector_database
from src.knowledge.zim_parser import DocumentChunk, ZIMParser


class RetrievedDocument:
    """Represents a retrieved document with relevance score."""
    
    def __init__(
        self,
        chunk: DocumentChunk,
        score: float,
    ):
        self.chunk = chunk
        self.score = score
        self.text = chunk.text
        self.metadata = chunk.metadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        return f"RetrievedDocument(score={self.score:.3f}, title={self.metadata.get('title', 'Unknown')})"


class RAGSystem:
    """
    Retrieval-Augmented Generation system.
    
    Provides knowledge retrieval capabilities for the agent system.
    """
    
    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        zim_parser: Optional[ZIMParser] = None,
    ):
        """
        Initialize RAG system.
        
        Args:
            vector_db: Vector database instance
            embedding_model: Embedding model instance
            zim_parser: ZIM parser instance
        """
        self.vector_db = vector_db or create_vector_database()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.zim_parser = zim_parser
        
        self.top_k = config.vector_db.vector_db_top_k
        
        logger.info("RAG system initialized")
    
    def index_documents(
        self,
        max_articles: Optional[int] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Index documents from ZIM file into vector database.
        
        Args:
            max_articles: Maximum number of articles to index
            batch_size: Number of chunks to process at once
            
        Returns:
            Total number of chunks indexed
        """
        if not self.zim_parser:
            logger.error("No ZIM parser available for indexing")
            return 0
        
        total_chunks = 0
        chunk_batch = []
        
        logger.info("Starting document indexing...")
        
        for article_chunks in self.zim_parser.parse_all_articles(max_articles):
            chunk_batch.extend(article_chunks)
            
            # Process batch when it reaches batch_size
            if len(chunk_batch) >= batch_size:
                self._index_chunk_batch(chunk_batch)
                total_chunks += len(chunk_batch)
                chunk_batch = []
                
                logger.info(f"Indexed {total_chunks} chunks so far...")
        
        # Process remaining chunks
        if chunk_batch:
            self._index_chunk_batch(chunk_batch)
            total_chunks += len(chunk_batch)
        
        logger.info(f"Indexing complete. Total chunks indexed: {total_chunks}")
        
        # Save database
        self.save_database()
        
        return total_chunks
    
    def _index_chunk_batch(self, chunks: List[DocumentChunk]) -> None:
        """
        Index a batch of chunks.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress=False)
        
        # Add to vector database
        self.vector_db.add_documents(chunks, embeddings)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievedDocument objects
        """
        top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, top_k=top_k)
        
        # Convert to RetrievedDocument objects
        retrieved_docs = [
            RetrievedDocument(chunk=chunk, score=score)
            for chunk, score in results
        ]
        
        logger.debug(
            f"Retrieved {len(retrieved_docs)} documents for query: '{query[:50]}...'"
        )
        
        return retrieved_docs
    
    def retrieve_multiple_queries(
        self,
        queries: List[str],
        top_k_per_query: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents for multiple queries and deduplicate.
        
        Args:
            queries: List of query strings
            top_k_per_query: Number of documents to retrieve per query
            
        Returns:
            Deduplicated list of RetrievedDocument objects
        """
        all_docs = []
        seen_chunk_ids = set()
        
        for query in queries:
            docs = self.retrieve(query, top_k=top_k_per_query)
            
            for doc in docs:
                if doc.chunk.chunk_id not in seen_chunk_ids:
                    all_docs.append(doc)
                    seen_chunk_ids.add(doc.chunk.chunk_id)
        
        # Sort by score
        all_docs.sort(key=lambda x: x.score, reverse=True)
        
        return all_docs
    
    def format_retrieved_documents(
        self,
        documents: List[RetrievedDocument],
        include_metadata: bool = True,
    ) -> str:
        """
        Format retrieved documents for display or input to LLM.
        
        Args:
            documents: List of RetrievedDocument objects
            include_metadata: Whether to include metadata in output
            
        Returns:
            Formatted string
        """
        if not documents:
            return "No relevant documents found."
        
        formatted_parts = []
        
        for i, doc in enumerate(documents, 1):
            part = f"[Document {i}]"
            
            if include_metadata:
                title = doc.metadata.get('title', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                part += f"\nTitle: {title}\nSource: {source}\nRelevance: {doc.score:.2f}"
            
            part += f"\n{doc.text}\n"
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
    
    def save_database(self) -> None:
        """Save the vector database to disk."""
        save_path = config.vector_db.vector_db_path
        self.vector_db.save(save_path)
        logger.info(f"Vector database saved to {save_path}")
    
    def load_database(self) -> None:
        """Load the vector database from disk."""
        load_path = config.vector_db.vector_db_path
        self.vector_db.load(load_path)
        logger.info(f"Vector database loaded from {load_path}")
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics."""
        return {
            "vector_db": self.vector_db.get_stats(),
            "embedding_dimension": self.embedding_model.get_dimension(),
            "top_k": self.top_k,
        }
