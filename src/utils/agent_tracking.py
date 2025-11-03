"""Agent activity tracking utilities."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional
import asyncio


@dataclass
class AgentRunRecord:
    agent_id: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class TaskActivity:
    task_id: str
    status: str = "created"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active_agents: Dict[str, AgentRunRecord] = field(default_factory=dict)
    history: Deque[AgentRunRecord] = field(default_factory=lambda: deque(maxlen=20))

    def to_overview(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "active_agents": [run.to_dict() for run in self.active_agents.values()],
            "recent_history": [record.to_dict() for record in list(self.history)],
        }


class AgentActivityTracker:
    """Tracks agent lifecycle events for monitoring."""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskActivity] = {}
        self._agent_stats: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=50))
        self._results: Dict[str, Dict[str, object]] = {}
        self._lock = asyncio.Lock()

    async def initialize_task(self, task_id: str) -> None:
        async with self._lock:
            self._tasks[task_id] = TaskActivity(task_id=task_id)

    async def update_task_status(self, task_id: str, status: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                task = TaskActivity(task_id=task_id)
                self._tasks[task_id] = task
            task.status = status
            task.updated_at = datetime.utcnow()

    async def agent_started(self, task_id: str, agent_id: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                task = TaskActivity(task_id=task_id)
                self._tasks[task_id] = task
            run = AgentRunRecord(
                agent_id=agent_id,
                status="running",
                started_at=datetime.utcnow(),
            )
            task.active_agents[agent_id] = run
            task.updated_at = datetime.utcnow()

    async def agent_completed(self, task_id: str, agent_id: str, duration_seconds: float) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            run = task.active_agents.pop(agent_id, None)
            if not run:
                run = AgentRunRecord(
                    agent_id=agent_id,
                    status="completed",
                    started_at=datetime.utcnow(),
                )
            run.status = "completed"
            run.ended_at = datetime.utcnow()
            run.duration_seconds = duration_seconds
            task.history.append(run)
            task.updated_at = datetime.utcnow()
            self._agent_stats[agent_id].append(duration_seconds)

    async def agent_failed(self, task_id: str, agent_id: str, error: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                task = TaskActivity(task_id=task_id)
                self._tasks[task_id] = task
            run = task.active_agents.pop(agent_id, None)
            if not run:
                run = AgentRunRecord(
                    agent_id=agent_id,
                    status="failed",
                    started_at=datetime.utcnow(),
                )
            run.status = "failed"
            run.ended_at = datetime.utcnow()
            run.error = error
            if run.duration_seconds is None:
                run.duration_seconds = (run.ended_at - run.started_at).total_seconds()
            task.history.append(run)
            task.updated_at = datetime.utcnow()

    async def finalize_task(self, task_id: str, status: str) -> None:
        await self.update_task_status(task_id, status)
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.active_agents.clear()

    async def get_task_activity(self, task_id: str) -> Dict[str, object]:
        async with self._lock:
            task = self._tasks.get(task_id)
            return task.to_overview() if task else {}

    async def get_overview(self) -> Dict[str, object]:
        async with self._lock:
            return {
                task_id: task.to_overview()
                for task_id, task in sorted(
                    self._tasks.items(),
                    key=lambda item: item[1].updated_at,
                    reverse=True,
                )
            }

    async def get_agent_stats(self) -> Dict[str, float]:
        async with self._lock:
            return {
                agent_id: (sum(durations) / len(durations))
                for agent_id, durations in self._agent_stats.items()
                if durations
            }

    async def set_result(self, task_id: str, result: Dict[str, object]) -> None:
        async with self._lock:
            self._results[task_id] = result

    async def get_result(self, task_id: str) -> Optional[Dict[str, object]]:
        async with self._lock:
            return self._results.get(task_id)


agent_tracker = AgentActivityTracker()
