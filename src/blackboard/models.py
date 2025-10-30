"""
Blackboard system models.
Defines data structures for blackboard entries and status.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class EntryType(str, Enum):
    """Blackboard entry type enumeration."""
    USER_INPUT = "UserInput"
    TASK_SUMMARY = "TaskSummary"
    KEYWORD = "Keyword"
    PLAN = "Plan"
    SUB_TASK = "SubTask"
    RETRIEVED_KNOWLEDGE = "RetrievedKnowledge"
    ANSWER_FORMAT = "AnswerFormat"
    SYNTHESIZED_DRAFT = "SynthesizedDraft"
    FINAL_ANSWER = "FinalAnswer"
    CONDUCTOR_DIRECTIVE = "ConductorDirective"
    ERROR = "Error"


class EntryMetadata(BaseModel):
    """Metadata for blackboard entries."""
    
    source: Optional[str] = Field(
        default=None,
        description="Source of the entry (e.g., 'User', 'RAG', 'LongTermMemory')"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the entry"
    )
    relevance_to: List[str] = Field(
        default_factory=list,
        description="List of entry IDs this entry is responding to"
    )
    additional_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class BlackboardEntry(BaseModel):
    """A single entry on the blackboard."""
    
    entry_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the entry"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the entry was created"
    )
    agent_id: str = Field(
        description="ID of the agent that created this entry"
    )
    entry_type: EntryType = Field(
        description="Type of the entry"
    )
    content: Any = Field(
        description="Actual content of the entry (text, JSON, etc.)"
    )
    metadata: EntryMetadata = Field(
        default_factory=EntryMetadata,
        description="Metadata associated with the entry"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class BlackboardState(BaseModel):
    """Current state of the blackboard."""
    
    task_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the task"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current status of the task"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the task was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    entries: List[BlackboardEntry] = Field(
        default_factory=list,
        description="List of all entries on the blackboard"
    )
    
    def update_status(self, new_status: TaskStatus) -> None:
        """Update the task status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.utcnow()
    
    def add_entry(self, entry: BlackboardEntry) -> None:
        """Add a new entry to the blackboard."""
        self.entries.append(entry)
        self.updated_at = datetime.utcnow()
    
    def get_entries_by_type(self, entry_type: EntryType) -> List[BlackboardEntry]:
        """Get all entries of a specific type."""
        return [e for e in self.entries if e.entry_type == entry_type]
    
    def get_entries_by_agent(self, agent_id: str) -> List[BlackboardEntry]:
        """Get all entries created by a specific agent."""
        return [e for e in self.entries if e.agent_id == agent_id]
    
    def get_latest_entry_by_type(self, entry_type: EntryType) -> Optional[BlackboardEntry]:
        """Get the most recent entry of a specific type."""
        entries = self.get_entries_by_type(entry_type)
        return entries[-1] if entries else None
    
    def get_entries_after_timestamp(self, timestamp: datetime) -> List[BlackboardEntry]:
        """Get all entries created after a specific timestamp."""
        return [e for e in self.entries if e.timestamp > timestamp]
    
    def get_entry_by_id(self, entry_id: str) -> Optional[BlackboardEntry]:
        """Get an entry by its ID."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        return None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
