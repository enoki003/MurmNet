"""
Orchestrator system for coordinating multiple agents.
Manages agent execution and emergent collaboration.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.agents import (
    LanguageModel,
    InputAnalyzerAgent,
    PlannerAgent,
    KnowledgeRetrieverAgent,
    ResponseFormatterAgent,
    SynthesizerAgent,
    ConductorAgent,
)
from src.blackboard import blackboard, EntryType, TaskStatus
from src.config import config
from src.config.agent_config import get_all_agent_definitions
from src.knowledge import RAGSystem
from src.memory import LongTermMemory, ExperienceMemory
from src.utils.agent_tracking import agent_tracker
from src.utils.metrics import performance_metrics


class Orchestrator:
    """
    Orchestrates the execution of multiple agents.
    
    Manages:
    - Agent initialization and lifecycle
    - Execution order and parallelism
    - Task completion detection
    - Experience recording
    """
    
    def __init__(
        self,
        rag_system: Optional[RAGSystem] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        experience_memory: Optional[ExperienceMemory] = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            rag_system: RAG system instance
            long_term_memory: Long-term memory instance
            experience_memory: Experience memory instance
        """
        self.rag_system = rag_system
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.experience_memory = experience_memory or ExperienceMemory()
        
        # Shared LLM instance for all agents
        self.llm = LanguageModel()
        
        # Initialize agents
        self.agents: Dict[str, Any] = {}
        self._initialize_agents()
        
        # Configuration
        self.max_parallel_agents = config.agent.max_parallel_agents
        self.max_iterations = 20  # Prevent infinite loops
        
        logger.info("Orchestrator initialized")
    
    def _initialize_agents(self) -> None:
        """Initialize all agents."""
        agent_definitions = get_all_agent_definitions()
        
        # Input Analyzer
        self.agents["input_analyzer"] = InputAnalyzerAgent(
            agent_definition=agent_definitions["input_analyzer"],
            blackboard=blackboard,
            llm=self.llm,
        )
        
        # Planner
        self.agents["planner"] = PlannerAgent(
            agent_definition=agent_definitions["planner"],
            blackboard=blackboard,
            llm=self.llm,
            experience_memory=self.experience_memory,
        )
        
        # Knowledge Retriever
        self.agents["knowledge_retriever"] = KnowledgeRetrieverAgent(
            agent_definition=agent_definitions["knowledge_retriever"],
            blackboard=blackboard,
            llm=self.llm,
            rag_system=self.rag_system,
            long_term_memory=self.long_term_memory,
        )
        
        # Response Formatter
        self.agents["response_formatter"] = ResponseFormatterAgent(
            agent_definition=agent_definitions["response_formatter"],
            blackboard=blackboard,
            llm=self.llm,
        )
        
        # Synthesizer
        self.agents["synthesizer"] = SynthesizerAgent(
            agent_definition=agent_definitions["synthesizer"],
            blackboard=blackboard,
            llm=self.llm,
        )
        
        # Conductor
        self.agents["conductor"] = ConductorAgent(
            agent_definition=agent_definitions["conductor"],
            blackboard=blackboard,
            llm=self.llm,
        )
        
        logger.info(f"Initialized {len(self.agents)} agents")

    async def _prepare_task(self) -> str:
        """Create blackboard task and initialize trackers."""
        state = await blackboard.create_task()
        task_id = state.task_id
        await agent_tracker.initialize_task(task_id)
        await agent_tracker.update_task_status(task_id, TaskStatus.PENDING.value)
        return task_id

    async def _run_task(self, task_id: str, user_query: str) -> Dict[str, Any]:
        """Execute the full task lifecycle for a query."""
        start_time = datetime.utcnow()

        logger.info(f"Processing query for task {task_id}: {user_query[:100]}...")

        try:
            # Write user input to blackboard
            await blackboard.write_entry(
                task_id=task_id,
                agent_id="system",
                entry_type=EntryType.USER_INPUT,
                content=user_query,
            )

            # Update status
            await blackboard.update_status(task_id, TaskStatus.ANALYZING)
            await agent_tracker.update_task_status(task_id, TaskStatus.ANALYZING.value)

            # Execute agents in coordination
            await self._execute_agent_swarm(task_id)

            # Get final answer
            final_answer_entries = await blackboard.read_entries(
                task_id=task_id,
                entry_type=EntryType.FINAL_ANSWER,
            )

            if not final_answer_entries:
                raise Exception("No final answer generated")

            final_answer = final_answer_entries[-1].content

            # Mark as completed
            await blackboard.update_status(task_id, TaskStatus.COMPLETED)
            await agent_tracker.finalize_task(task_id, TaskStatus.COMPLETED.value)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Store experience
            await self._store_experience(task_id, user_query, final_answer, True, execution_time)

            # Store important information in long-term memory
            await self._update_long_term_memory(task_id, user_query, final_answer)

            logger.info(f"Query processed successfully in {execution_time:.2f}s")
            await performance_metrics.record_query(task_id, True, execution_time)

            result: Dict[str, Any] = {
                "task_id": task_id,
                "query": user_query,
                "answer": final_answer,
                "execution_time_seconds": execution_time,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Query processing failed: {e}")

            # Mark as failed
            await blackboard.update_status(task_id, TaskStatus.FAILED)
            await agent_tracker.finalize_task(task_id, TaskStatus.FAILED.value)

            # Store failed experience
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self._store_experience(
                task_id, user_query, f"Error: {str(e)}", False, execution_time
            )
            await performance_metrics.record_query(task_id, False, execution_time)

            result = {
                "task_id": task_id,
                "query": user_query,
                "error": str(e),
                "execution_time_seconds": execution_time,
                "success": False,
            }

        await agent_tracker.set_result(task_id, result)
        return result

    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a query synchronously and return the result."""
        task_id = await self._prepare_task()
        return await self._run_task(task_id, user_query)

    async def start_query(self, user_query: str) -> str:
        """Start processing a query in the background and return the task id."""
        task_id = await self._prepare_task()

        async def runner() -> None:
            try:
                await self._run_task(task_id, user_query)
            except Exception:
                # Errors are handled inside _run_task; we silence propagation here.
                logger.exception("Background query failed")

        asyncio.create_task(runner())
        return task_id
    
    async def _execute_agent_swarm(self, task_id: str) -> None:
        """
        Execute agent swarm with emergent coordination.
        
        Args:
            task_id: Task identifier
        """
        # Define execution phases
        phases = [
            {
                "name": "Analysis",
                "agents": ["input_analyzer"],
                "status": TaskStatus.ANALYZING,
            },
            {
                "name": "Planning",
                "agents": ["planner", "response_formatter"],
                "status": TaskStatus.PLANNING,
            },
            {
                "name": "Retrieval",
                "agents": ["knowledge_retriever"],
                "status": TaskStatus.RETRIEVING,
            },
            {
                "name": "Synthesis",
                "agents": ["synthesizer"],
                "status": TaskStatus.SYNTHESIZING,
            },
        ]
        
        for phase in phases:
            logger.info(f"Entering phase: {phase['name']}")
            
            # Update status
            await blackboard.update_status(task_id, phase["status"])
            await agent_tracker.update_task_status(task_id, phase["status"].value)
            
            # Execute agents in this phase (potentially in parallel)
            agent_tasks = []
            for agent_id in phase["agents"]:
                agent = self.agents.get(agent_id)
                if agent and await agent.can_execute(task_id):
                    agent_tasks.append(agent.execute(task_id))
            
            # Execute agents
            if agent_tasks:
                results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                
                # Log any errors
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Agent failed in {phase['name']}: {result}")
        
        logger.info("Agent swarm execution completed")
    
    async def _store_experience(
        self,
        task_id: str,
        problem: str,
        outcome: str,
        success: bool,
        execution_time: float,
    ) -> None:
        """
        Store task execution as experience.
        
        Args:
            task_id: Task identifier
            problem: Problem description
            outcome: Outcome of execution
            success: Whether successful
            execution_time: Execution time in seconds
        """
        try:
            # Get plan from blackboard
            plan_entries = await blackboard.read_entries(
                task_id=task_id,
                entry_type=EntryType.PLAN,
            )
            
            plan = plan_entries[-1].content if plan_entries else {}
            
            # Get agent sequence
            summary = await blackboard.get_task_summary(task_id)
            agent_sequence = list(summary.get("agent_activity", {}).keys())
            
            # Generate lessons learned
            lessons = await self._generate_lessons_learned(task_id, success)
            
            # Store in experience memory
            self.experience_memory.store_experience(
                task_id=task_id,
                problem=problem,
                plan=plan,
                outcome=outcome,
                success=success,
                agent_sequence=agent_sequence,
                execution_time_seconds=execution_time,
                lessons_learned=lessons,
            )
            
            logger.debug(f"Stored experience for task {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
    
    async def _generate_lessons_learned(self, task_id: str, success: bool) -> str:
        """
        Generate lessons learned from task execution.
        
        Args:
            task_id: Task identifier
            success: Whether task succeeded
            
        Returns:
            Lessons learned text
        """
        if success:
            return "タスクは成功しました。エージェント間の協調がうまく機能しました。"
        else:
            return "タスクは失敗しました。計画の見直しやより詳細な情報検索が必要かもしれません。"
    
    async def _update_long_term_memory(
        self,
        task_id: str,
        query: str,
        answer: str,
    ) -> None:
        """
        Update long-term memory with important information.
        
        Args:
            task_id: Task identifier
            query: User query
            answer: Generated answer
        """
        try:
            # Create memory summary
            memory_content = f"質問: {query}\n\n回答: {answer}"
            
            # Store in long-term memory with moderate importance
            self.long_term_memory.store_memory(
                content=memory_content,
                metadata={"task_id": task_id, "type": "qa_pair"},
                importance_score=0.6,
            )
            
            logger.debug(f"Updated long-term memory for task {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to update long-term memory: {e}")
    
    async def get_task_history(self, task_id: str) -> Optional[Dict]:
        """
        Get complete history of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task history dict
        """
        return await blackboard.export_task_history(task_id)
    
    def get_statistics(self) -> Dict:
        """
        Get system statistics.
        
        Returns:
            Dict with system statistics
        """
        stats = {
            "agents": len(self.agents),
            "experience_memory": self.experience_memory.get_statistics(),
            "long_term_memory_count": self.long_term_memory.get_memory_count(),
        }
        
        if self.rag_system:
            stats["rag_system"] = self.rag_system.get_stats()
        
        return stats
