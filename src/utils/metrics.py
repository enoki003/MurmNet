"""Performance metrics collection utilities."""

from collections import deque
from datetime import datetime
from typing import Deque, Dict, List
import asyncio


class PerformanceMetrics:
    """Aggregates query-level performance statistics."""

    def __init__(self, max_history: int = 100) -> None:
        self._history: Deque[Dict[str, object]] = deque(maxlen=max_history)
        self._lock = asyncio.Lock()

    async def record_query(
        self,
        task_id: str,
        success: bool,
        execution_time_seconds: float,
    ) -> None:
        async with self._lock:
            self._history.appendleft(
                {
                    "task_id": task_id,
                    "success": success,
                    "execution_time_seconds": execution_time_seconds,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    async def get_summary(self) -> Dict[str, object]:
        async with self._lock:
            history = list(self._history)
        if not history:
            return {
                "total_queries": 0,
                "success_rate": 0.0,
                "average_execution_time_seconds": 0.0,
                "recent": [],
            }
        total = len(history)
        successes = sum(1 for item in history if item["success"])
        avg_time = sum(item["execution_time_seconds"] for item in history) / total
        return {
            "total_queries": total,
            "success_rate": successes / total,
            "average_execution_time_seconds": avg_time,
            "recent": history[:10],
        }


performance_metrics = PerformanceMetrics()
