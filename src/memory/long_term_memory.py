"""
Long-term memory system.
Stores and retrieves important information from past conversations.
"""

import json
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.config import config
from src.knowledge.embeddings import EmbeddingModel
from src.knowledge.vector_db import VectorDatabase, create_vector_database
from src.knowledge.zim_parser import DocumentChunk
from src.memory.models import MemoryEntry


class LongTermMemory:
    """
    Long-term memory system for storing important information.
    
    Uses vector similarity search to retrieve relevant memories.
    """
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize long-term memory system.
        
        Args:
            embedding_model: Embedding model for vectorization
            storage_path: Path to store memory data
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_path = storage_path or config.memory.memory_db_path / "long_term"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create separate vector database for memories
        self.vector_db: VectorDatabase = create_vector_database()
        
        # In-memory storage for full memory entries
        self.memories: List[MemoryEntry] = []
        
        self.top_k = config.memory.long_term_memory_top_k
        
        logger.info("Long-term memory system initialized")
        
        # Try to load existing memories
        self._load_memories()
    
    def store_memory(
        self,
        content: str,
        metadata: Optional[dict] = None,
        importance_score: float = 0.5,
    ) -> MemoryEntry:
        """
        Store a new memory.
        
        Args:
            content: Content to remember
            metadata: Additional metadata
            importance_score: Importance score (0-1)
            
        Returns:
            Created MemoryEntry
        """
        # Create memory entry
        memory = MemoryEntry(
            content=content,
            metadata=metadata or {},
            importance_score=importance_score,
        )
        
        # Generate embedding
        embedding = self.embedding_model.encode_single(content)
        memory.embedding = embedding
        
        # Add to memories list
        self.memories.append(memory)
        
        # Create document chunk for vector DB
        chunk = DocumentChunk(
            chunk_id=memory.memory_id,
            text=content,
            metadata={
                "type": "long_term_memory",
                "importance": str(importance_score),
                **memory.metadata,
            },
        )
        
        # Add to vector database
        self.vector_db.add_documents([chunk], [embedding])
        
        logger.debug(f"Stored memory: {memory.memory_id}")
        
        # Persist to disk
        self._save_memories()
        
        return memory
    
    def retrieve_memories(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: Query string
            top_k: Number of memories to retrieve
            min_importance: Minimum importance score threshold
            
        Returns:
            List of relevant MemoryEntry objects
        """
        top_k = top_k or self.top_k
        
        if not self.memories:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, top_k=top_k * 2)
        
        # Convert to MemoryEntry objects and filter
        retrieved_memories = []
        for chunk, score in results:
            # Find corresponding memory
            memory = self._get_memory_by_id(chunk.chunk_id)
            if memory and memory.importance_score >= min_importance:
                # Update access statistics
                memory.access_count += 1
                memory.last_accessed = datetime.utcnow()
                
                retrieved_memories.append(memory)
                
                if len(retrieved_memories) >= top_k:
                    break
        
        logger.debug(f"Retrieved {len(retrieved_memories)} memories for query")
        
        return retrieved_memories
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get memory by ID."""
        for memory in self.memories:
            if memory.memory_id == memory_id:
                return memory
        return None
    
    def get_all_memories(self) -> List[MemoryEntry]:
        """Get all stored memories."""
        return self.memories.copy()
    
    def get_memory_count(self) -> int:
        """Get total number of memories."""
        return len(self.memories)
    
    def _save_memories(self) -> None:
        """Save memories to disk."""
        memories_file = self.storage_path / "memories.json"
        
        try:
            # Convert memories to dict
            memories_data = [
                memory.dict() for memory in self.memories
            ]
            
            with open(memories_file, 'w', encoding='utf-8') as f:
                json.dump(memories_data, f, ensure_ascii=False, indent=2)
            
            # Save vector database
            self.vector_db.save(self.storage_path / "vector_db")
            
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
    
    def _load_memories(self) -> None:
        """Load memories from disk."""
        memories_file = self.storage_path / "memories.json"
        vector_db_path = self.storage_path / "vector_db"
        
        if not memories_file.exists():
            logger.info("No existing memories found")
            return
        
        try:
            # Load memories
            with open(memories_file, 'r', encoding='utf-8') as f:
                memories_data = json.load(f)
            
            self.memories = [
                MemoryEntry(**data) for data in memories_data
            ]
            
            # Load vector database
            if vector_db_path.exists():
                self.vector_db.load(vector_db_path)
            
            logger.info(f"Loaded {len(self.memories)} memories from disk")
            
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            self.memories = []
    
    def format_memories(self, memories: List[MemoryEntry]) -> str:
        """
        Format memories for display or input to LLM.
        
        Args:
            memories: List of MemoryEntry objects
            
        Returns:
            Formatted string
        """
        if not memories:
            return "No relevant memories found."
        
        formatted_parts = []
        
        for i, memory in enumerate(memories, 1):
            part = f"[Memory {i}]"
            part += f"\n{memory.content}"
            part += f"\n(Importance: {memory.importance_score:.2f}, "
            part += f"Accessed: {memory.access_count} times)\n"
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)


# Import datetime for access tracking
from datetime import datetime
