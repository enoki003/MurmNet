"""
Memory system models.
Defines data structures for long-term memory and experience memory.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single memory entry."""
    
    memory_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the memory"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the memory was created"
    )
    content: str = Field(
        description="Content of the memory"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding of the content"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    importance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score of the memory"
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this memory has been accessed"
    )
    last_accessed: Optional[datetime] = Field(
        default=None,
        description="Last time this memory was accessed"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ExperienceEntry(BaseModel):
    """An experience entry from a completed task."""
    
    experience_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the experience"
    )
    task_id: str = Field(
        description="ID of the task this experience came from"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the experience was created"
    )
    problem: str = Field(
        description="Description of the problem/task"
    )
    plan: Dict[str, Any] = Field(
        description="The plan that was executed"
    )
    outcome: str = Field(
        description="Result of the task execution"
    )
    success: bool = Field(
        description="Whether the task was successful"
    )
    agent_sequence: List[str] = Field(
        default_factory=list,
        description="Sequence of agents involved"
    )
    execution_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to complete the task"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding of the problem"
    )
    lessons_learned: Optional[str] = Field(
        default=None,
        description="Key lessons from this experience"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
