"""
Experience memory system.
Stores and retrieves past task execution experiences.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.config import config
from src.knowledge.embeddings import EmbeddingModel
from src.knowledge.vector_db import VectorDatabase, create_vector_database
from src.knowledge.zim_parser import DocumentChunk
from src.memory.models import ExperienceEntry


class ExperienceMemory:
    """
    Experience memory system for storing task execution patterns.
    
    Learns from successful and failed attempts to improve future performance.
    """
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize experience memory system.
        
        Args:
            embedding_model: Embedding model for vectorization
            storage_path: Path to store experience data
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.storage_path = storage_path or config.memory.memory_db_path / "experience"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create separate vector database for experiences
        self.vector_db: VectorDatabase = create_vector_database()
        
        # In-memory storage for full experience entries
        self.experiences: List[ExperienceEntry] = []
        
        self.top_k = config.memory.experience_memory_top_k
        
        logger.info("Experience memory system initialized")
        
        # Try to load existing experiences
        self._load_experiences()
    
    def store_experience(
        self,
        task_id: str,
        problem: str,
        plan: Dict[str, Any],
        outcome: str,
        success: bool,
        agent_sequence: Optional[List[str]] = None,
        execution_time_seconds: float = 0.0,
        lessons_learned: Optional[str] = None,
    ) -> ExperienceEntry:
        """
        Store a new experience.
        
        Args:
            task_id: Task identifier
            problem: Description of the problem
            plan: The plan that was executed
            outcome: Result of execution
            success: Whether the task succeeded
            agent_sequence: Sequence of agents involved
            execution_time_seconds: Execution time
            lessons_learned: Key lessons from this experience
            
        Returns:
            Created ExperienceEntry
        """
        # Create experience entry
        experience = ExperienceEntry(
            task_id=task_id,
            problem=problem,
            plan=plan,
            outcome=outcome,
            success=success,
            agent_sequence=agent_sequence or [],
            execution_time_seconds=execution_time_seconds,
            lessons_learned=lessons_learned,
        )
        
        # Generate embedding from problem description
        embedding = self.embedding_model.encode_single(problem)
        experience.embedding = embedding
        
        # Add to experiences list
        self.experiences.append(experience)
        
        # Create document chunk for vector DB
        chunk = DocumentChunk(
            chunk_id=experience.experience_id,
            text=problem,
            metadata={
                "type": "experience",
                "task_id": task_id,
                "success": str(success),
                "execution_time": str(execution_time_seconds),
            },
        )
        
        # Add to vector database
        self.vector_db.add_documents([chunk], [embedding])
        
        logger.debug(
            f"Stored experience: {experience.experience_id} "
            f"(success={success})"
        )
        
        # Persist to disk
        self._save_experiences()
        
        return experience
    
    def retrieve_similar_experiences(
        self,
        problem: str,
        top_k: Optional[int] = None,
        success_only: bool = False,
    ) -> List[ExperienceEntry]:
        """
        Retrieve similar past experiences.
        
        Args:
            problem: Problem description to search for
            top_k: Number of experiences to retrieve
            success_only: Only return successful experiences
            
        Returns:
            List of relevant ExperienceEntry objects
        """
        top_k = top_k or self.top_k
        
        if not self.experiences:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(problem)
        
        # Search vector database
        search_limit = top_k * 3 if success_only else top_k
        results = self.vector_db.search(query_embedding, top_k=search_limit)
        
        # Convert to ExperienceEntry objects and filter
        retrieved_experiences = []
        for chunk, score in results:
            # Find corresponding experience
            experience = self._get_experience_by_id(chunk.chunk_id)
            if experience:
                # Filter by success if requested
                if success_only and not experience.success:
                    continue
                
                retrieved_experiences.append(experience)
                
                if len(retrieved_experiences) >= top_k:
                    break
        
        logger.debug(
            f"Retrieved {len(retrieved_experiences)} experiences "
            f"(success_only={success_only})"
        )
        
        return retrieved_experiences
    
    def _get_experience_by_id(self, experience_id: str) -> Optional[ExperienceEntry]:
        """Get experience by ID."""
        for experience in self.experiences:
            if experience.experience_id == experience_id:
                return experience
        return None
    
    def get_all_experiences(self) -> List[ExperienceEntry]:
        """Get all stored experiences."""
        return self.experiences.copy()
    
    def get_experience_count(self) -> int:
        """Get total number of experiences."""
        return len(self.experiences)
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.experiences:
            return 0.0
        
        successful = sum(1 for exp in self.experiences if exp.success)
        return successful / len(self.experiences)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored experiences."""
        if not self.experiences:
            return {
                "total_experiences": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
            }
        
        successful = sum(1 for exp in self.experiences if exp.success)
        total_time = sum(exp.execution_time_seconds for exp in self.experiences)
        
        return {
            "total_experiences": len(self.experiences),
            "successful_experiences": successful,
            "failed_experiences": len(self.experiences) - successful,
            "success_rate": successful / len(self.experiences),
            "average_execution_time": total_time / len(self.experiences),
        }
    
    def _save_experiences(self) -> None:
        """Save experiences to disk."""
        experiences_file = self.storage_path / "experiences.json"
        
        try:
            # Convert experiences to dict with datetime handling
            experiences_data = []
            for experience in self.experiences:
                exp_dict = experience.model_dump(mode="json")
                experiences_data.append(exp_dict)
            
            with open(experiences_file, 'w', encoding='utf-8') as f:
                json.dump(experiences_data, f, ensure_ascii=False, indent=2)
            
            # Save vector database
            self.vector_db.save(self.storage_path / "vector_db")
            
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")
    
    def _load_experiences(self) -> None:
        """Load experiences from disk."""
        experiences_file = self.storage_path / "experiences.json"
        vector_db_path = self.storage_path / "vector_db"
        
        if not experiences_file.exists():
            logger.info("No existing experiences found")
            return
        
        try:
            # Load experiences
            with open(experiences_file, 'r', encoding='utf-8') as f:
                experiences_data = json.load(f)
            
            self.experiences = [
                ExperienceEntry(**data) for data in experiences_data
            ]
            
            # Load vector database
            if vector_db_path.exists():
                self.vector_db.load(vector_db_path)
            
            logger.info(f"Loaded {len(self.experiences)} experiences from disk")
            
        except Exception as e:
            logger.error(f"Failed to load experiences: {e}")
            self.experiences = []
    
    def format_experiences(
        self,
        experiences: List[ExperienceEntry],
        include_plan: bool = True,
    ) -> str:
        """
        Format experiences for display or input to LLM.
        
        Args:
            experiences: List of ExperienceEntry objects
            include_plan: Whether to include full plan details
            
        Returns:
            Formatted string
        """
        if not experiences:
            return "No relevant experiences found."
        
        formatted_parts = []
        
        for i, exp in enumerate(experiences, 1):
            part = f"[Experience {i}]"
            part += f"\nProblem: {exp.problem}"
            
            if include_plan:
                plan_summary = json.dumps(exp.plan, ensure_ascii=False, indent=2)
                part += f"\nPlan: {plan_summary}"
            
            part += f"\nOutcome: {exp.outcome}"
            part += f"\nSuccess: {'Yes' if exp.success else 'No'}"
            part += f"\nExecution Time: {exp.execution_time_seconds:.2f}s"
            
            if exp.lessons_learned:
                part += f"\nLessons: {exp.lessons_learned}"
            
            part += "\n"
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
