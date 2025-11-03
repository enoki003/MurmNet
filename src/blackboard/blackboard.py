"""
Blackboard system implementation.
Central workspace for agent collaboration and information sharing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.blackboard.models import (
    BlackboardEntry,
    BlackboardState,
    EntryMetadata,
    EntryType,
    TaskStatus,
)
from src.config import config


class BlackboardSystem:
    """
    Central blackboard system for agent collaboration.
    
    This system maintains a shared workspace where agents can read and write
    information, enabling emergent collaborative intelligence.
    """
    
    def __init__(self):
        self._states: Dict[str, BlackboardState] = {}
        self._observers: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
        
        # Configuration
        self._max_entries = config.blackboard.blackboard_max_entries
        self._retention_hours = config.blackboard.blackboard_retention_hours
        
        logger.info("Blackboard system initialized")
    
    async def create_task(self, task_id: Optional[str] = None) -> BlackboardState:
        """
        Create a new task with an associated blackboard.
        
        Args:
            task_id: Optional custom task ID
            
        Returns:
            New BlackboardState instance
        """
        async with self._lock:
            state = BlackboardState()
            if task_id:
                state.task_id = task_id
            
            self._states[state.task_id] = state
            self._observers[state.task_id] = []
            
            logger.info(f"Created new task: {state.task_id}")
            return state
    
    async def get_state(self, task_id: str) -> Optional[BlackboardState]:
        """
        Get the current state of a task's blackboard.
        
        Args:
            task_id: Task identifier
            
        Returns:
            BlackboardState if exists, None otherwise
        """
        return self._states.get(task_id)
    
    async def write_entry(
        self,
        task_id: str,
        agent_id: str,
        entry_type: EntryType,
        content: Any,
        metadata: Optional[EntryMetadata] = None,
    ) -> BlackboardEntry:
        """
        Write a new entry to the blackboard.
        
        Args:
            task_id: Task identifier
            agent_id: ID of the agent writing the entry
            entry_type: Type of the entry
            content: Content to write
            metadata: Optional metadata
            
        Returns:
            Created BlackboardEntry
            
        Raises:
            ValueError: If task_id doesn't exist
        """
        async with self._lock:
            state = self._states.get(task_id)
            if not state:
                raise ValueError(f"Task {task_id} not found")
            
            # Create entry
            entry = BlackboardEntry(
                agent_id=agent_id,
                entry_type=entry_type,
                content=content,
                metadata=metadata or EntryMetadata(),
            )
            
            # Add to state
            state.add_entry(entry)
            
            # Enforce max entries limit
            if len(state.entries) > self._max_entries:
                removed_count = len(state.entries) - self._max_entries
                state.entries = state.entries[-self._max_entries:]
                logger.warning(
                    f"Blackboard for task {task_id} exceeded max entries. "
                    f"Removed {removed_count} oldest entries."
                )
            
            logger.debug(
                f"Entry written to blackboard - Task: {task_id}, "
                f"Agent: {agent_id}, Type: {entry_type.value}"
            )
            
            # Notify observers
            await self._notify_observers(task_id, entry)
            
            return entry
    
    async def read_entries(
        self,
        task_id: str,
        entry_type: Optional[EntryType] = None,
        agent_id: Optional[str] = None,
        after_timestamp: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[BlackboardEntry]:
        """
        Read entries from the blackboard with optional filtering.
        
        Args:
            task_id: Task identifier
            entry_type: Filter by entry type
            agent_id: Filter by agent ID
            after_timestamp: Only return entries after this timestamp
            limit: Maximum number of entries to return
            
        Returns:
            List of matching BlackboardEntry objects
        """
        state = self._states.get(task_id)
        if not state:
            return []
        
        entries = state.entries
        
        # Apply filters
        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]
        
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        
        if after_timestamp:
            entries = [e for e in entries if e.timestamp > after_timestamp]
        
        # Apply limit
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    async def update_status(self, task_id: str, status: TaskStatus) -> None:
        """
        Update the status of a task.
        
        Args:
            task_id: Task identifier
            status: New status
            
        Raises:
            ValueError: If task_id doesn't exist
        """
        async with self._lock:
            state = self._states.get(task_id)
            if not state:
                raise ValueError(f"Task {task_id} not found")
            
            old_status = state.status
            state.update_status(status)
            
            logger.info(
                f"Task {task_id} status updated: {old_status.value} -> {status.value}"
            )
    
    async def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskStatus if task exists, None otherwise
        """
        state = self._states.get(task_id)
        return state.status if state else None
    
    async def register_observer(
        self,
        task_id: str,
        callback: Callable[[BlackboardEntry], None],
    ) -> None:
        """
        Register an observer callback for blackboard changes.
        
        Args:
            task_id: Task identifier
            callback: Async callback function to be called on new entries
        """
        if task_id not in self._observers:
            self._observers[task_id] = []
        
        self._observers[task_id].append(callback)
        logger.debug(f"Registered observer for task {task_id}")
    
    async def _notify_observers(self, task_id: str, entry: BlackboardEntry) -> None:
        """
        Notify all observers of a new entry.
        
        Args:
            task_id: Task identifier
            entry: New BlackboardEntry
        """
        observers = self._observers.get(task_id, [])
        
        for callback in observers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(entry)
                else:
                    callback(entry)
            except Exception as e:
                logger.error(f"Error in observer callback: {e}")
    
    async def cleanup_old_tasks(self) -> int:
        """
        Remove tasks older than retention period.
        
        Returns:
            Number of tasks removed
        """
        async with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(
                hours=self._retention_hours
            )
            
            tasks_to_remove = [
                task_id
                for task_id, state in self._states.items()
                if state.updated_at < cutoff_time
            ]
            
            for task_id in tasks_to_remove:
                del self._states[task_id]
                if task_id in self._observers:
                    del self._observers[task_id]
            
            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
            
            return len(tasks_to_remove)
    
    async def get_task_summary(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of the task's blackboard state.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dictionary containing task summary
        """
        state = self._states.get(task_id)
        if not state:
            return None
        
        entry_type_counts = {}
        for entry in state.entries:
            entry_type_str = entry.entry_type.value
            entry_type_counts[entry_type_str] = (
                entry_type_counts.get(entry_type_str, 0) + 1
            )
        
        agent_counts = {}
        for entry in state.entries:
            agent_counts[entry.agent_id] = agent_counts.get(entry.agent_id, 0) + 1
        
        return {
            "task_id": task_id,
            "status": state.status.value,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "total_entries": len(state.entries),
            "entry_types": entry_type_counts,
            "agent_activity": agent_counts,
        }

    async def get_all_task_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries for all tasks."""
        async with self._lock:
            states = list(self._states.values())

        summaries: List[Dict[str, Any]] = []
        for state in states:
            entry_type_counts: Dict[str, int] = {}
            agent_counts: Dict[str, int] = {}

            for entry in state.entries:
                entry_type_str = entry.entry_type.value
                entry_type_counts[entry_type_str] = (
                    entry_type_counts.get(entry_type_str, 0) + 1
                )
                agent_counts[entry.agent_id] = agent_counts.get(entry.agent_id, 0) + 1

            summaries.append(
                {
                    "task_id": state.task_id,
                    "status": state.status.value,
                    "created_at": state.created_at.isoformat(),
                    "updated_at": state.updated_at.isoformat(),
                    "total_entries": len(state.entries),
                    "entry_types": entry_type_counts,
                    "agent_activity": agent_counts,
                }
            )

        summaries.sort(key=lambda item: item["updated_at"], reverse=True)
        return summaries
    
    async def export_task_history(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Export the complete history of a task for analysis.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dictionary containing complete task history
        """
        state = self._states.get(task_id)
        if not state:
            return None
        
        return {
            "task_id": state.task_id,
            "status": state.status.value,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "timestamp": entry.timestamp.isoformat(),
                    "agent_id": entry.agent_id,
                    "entry_type": entry.entry_type.value,
                    "content": entry.content,
                    "metadata": entry.metadata.dict(),
                }
                for entry in state.entries
            ],
        }


# Global blackboard instance
blackboard = BlackboardSystem()
