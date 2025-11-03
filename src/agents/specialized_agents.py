"""
Specialized agent implementations.
Each agent has a specific role in the collaborative system.
"""

from typing import Any, Dict, List

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.blackboard import EntryType
from src.knowledge import RAGSystem
from src.memory import LongTermMemory, ExperienceMemory


class InputAnalyzerAgent(BaseAgent):
    """
    Analyzes user input and extracts key information.
    """
    
    async def _execute_impl(self, task_id: str) -> List[Dict[str, Any]]:
        """Execute input analysis."""
        # Read user input
        user_inputs = await self._read_blackboard(task_id, EntryType.USER_INPUT)
        
        if not user_inputs:
            logger.warning("No user input found")
            return []
        
        user_input = user_inputs[-1].content
        
        # Prepare prompt
        messages = self._format_prompt(
            user_content=user_input,
            context={"user_input": user_input},
        )
        
        # Generate analysis
        response = await self._generate_response(messages)
        
        # Parse JSON response
        analysis = self._parse_json_response(response)
        
        # Write task summary
        await self._write_blackboard(
            task_id=task_id,
            entry_type=EntryType.TASK_SUMMARY,
            content=analysis.get("task_summary", ""),
        )
        
        # Write keywords
        await self._write_blackboard(
            task_id=task_id,
            entry_type=EntryType.KEYWORD,
            content=analysis.get("keywords", []),
        )
        
        return [
            {"type": "task_summary", "content": analysis.get("task_summary")},
            {"type": "keywords", "content": analysis.get("keywords")},
        ]


class PlannerAgent(BaseAgent):
    """
    Creates step-by-step plans to solve tasks.
    """
    
    def __init__(self, *args, experience_memory: ExperienceMemory = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.experience_memory = experience_memory
    
    async def _execute_impl(self, task_id: str) -> List[Dict[str, Any]]:
        """Execute planning."""
        # Read task summary and keywords
        summaries = await self._read_blackboard(task_id, EntryType.TASK_SUMMARY)
        keywords = await self._read_blackboard(task_id, EntryType.KEYWORD)
        
        if not summaries:
            logger.warning("No task summary found")
            return []
        
        task_summary = summaries[-1].content
        keyword_list = keywords[-1].content if keywords else []
        
        # Check experience memory for similar tasks
        similar_experiences_text = ""
        if self.experience_memory:
            similar_experiences = self.experience_memory.retrieve_similar_experiences(
                problem=task_summary,
                top_k=2,
                success_only=True,
            )
            if similar_experiences:
                similar_experiences_text = "\n\n参考となる過去の経験:\n"
                similar_experiences_text += self.experience_memory.format_experiences(
                    similar_experiences,
                    include_plan=True,
                )
        
        # Prepare prompt with experience context included in the main content
        context = {
            "task_summary": task_summary,
            "keywords": ", ".join(keyword_list) if isinstance(keyword_list, list) else str(keyword_list),
        }
        
        # Combine task summary with experience context
        user_content = task_summary
        if similar_experiences_text:
            user_content += similar_experiences_text
        
        messages = self._format_prompt(
            user_content=user_content,
            context=context,
        )
        
        # Generate plan
        response = await self._generate_response(messages)
        
        # Parse JSON response
        plan_data = self._parse_json_response(response)
        
        # Write plan
        await self._write_blackboard(
            task_id=task_id,
            entry_type=EntryType.PLAN,
            content=plan_data.get("plan", {}),
        )
        
        return [{"type": "plan", "content": plan_data.get("plan")}]


class KnowledgeRetrieverAgent(BaseAgent):
    """
    Retrieves relevant knowledge from RAG system and memory.
    """
    
    def __init__(
        self,
        *args,
        rag_system: RAGSystem = None,
        long_term_memory: LongTermMemory = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rag_system = rag_system
        self.long_term_memory = long_term_memory
    
    async def _execute_impl(self, task_id: str) -> List[Dict[str, Any]]:
        """Execute knowledge retrieval."""
        # Read keywords and plan
        keywords = await self._read_blackboard(task_id, EntryType.KEYWORD)
        plans = await self._read_blackboard(task_id, EntryType.PLAN)
        
        if not keywords:
            logger.warning("No keywords found")
            return []
        
        keyword_list = keywords[-1].content
        if not isinstance(keyword_list, list):
            keyword_list = [str(keyword_list)]
        
        # Get plan context if available
        plan_context = ""
        if plans:
            plan_data = plans[-1].content
            if isinstance(plan_data, dict) and "steps" in plan_data:
                steps = plan_data["steps"]
                plan_context = "計画のステップ:\n"
                for step in steps:
                    if isinstance(step, dict):
                        plan_context += f"- {step.get('action', '')}\n"
        
        # Retrieve from RAG system
        rag_docs = []
        if self.rag_system:
            try:
                rag_docs = self.rag_system.retrieve_multiple_queries(
                    queries=keyword_list,
                    top_k_per_query=3,
                )
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")
        
        # Retrieve from long-term memory
        memory_entries = []
        if self.long_term_memory:
            for keyword in keyword_list:
                memories = self.long_term_memory.retrieve_memories(
                    query=keyword,
                    top_k=2,
                )
                memory_entries.extend(memories)
        
        # Format retrieved knowledge
        retrieved_knowledge_text = ""
        
        if rag_docs:
            retrieved_knowledge_text += "【知識ベースから】\n"
            retrieved_knowledge_text += self.rag_system.format_retrieved_documents(
                rag_docs[:5],
                include_metadata=True,
            )
        
        if memory_entries:
            retrieved_knowledge_text += "\n\n【長期記憶から】\n"
            retrieved_knowledge_text += self.long_term_memory.format_memories(
                memory_entries[:3]
            )
        
        if not retrieved_knowledge_text:
            retrieved_knowledge_text = "関連する知識が見つかりませんでした。"
        
        # Prepare context for LLM to organize the knowledge
        context = {
            "query": ", ".join(keyword_list),
            "plan_context": plan_context,
            "retrieved_knowledge": retrieved_knowledge_text,
        }
        
        messages = self._format_prompt(
            user_content=retrieved_knowledge_text,
            context=context,
        )
        
        # Generate organized knowledge
        response = await self._generate_response(messages)
        
        # Parse JSON response
        organized_knowledge = self._parse_json_response(response)
        
        # Write retrieved knowledge
        await self._write_blackboard(
            task_id=task_id,
            entry_type=EntryType.RETRIEVED_KNOWLEDGE,
            content=organized_knowledge,
        )
        
        return [{"type": "retrieved_knowledge", "content": organized_knowledge}]


class ResponseFormatterAgent(BaseAgent):
    """
    Determines the optimal format for the response.
    """
    
    async def _execute_impl(self, task_id: str) -> List[Dict[str, Any]]:
        """Execute response format specification."""
        # Read task summary and user input
        summaries = await self._read_blackboard(task_id, EntryType.TASK_SUMMARY)
        user_inputs = await self._read_blackboard(task_id, EntryType.USER_INPUT)
        
        if not summaries:
            logger.warning("No task summary found")
            return []
        
        task_summary = summaries[-1].content
        user_input = user_inputs[-1].content if user_inputs else ""
        
        # Prepare prompt
        context = {
            "task_summary": task_summary,
            "user_input": user_input,
        }
        
        messages = self._format_prompt(
            user_content=task_summary,
            context=context,
        )
        
        # Generate format specification
        response = await self._generate_response(messages)
        
        # Parse JSON response
        format_spec = self._parse_json_response(response)
        
        # Write answer format
        await self._write_blackboard(
            task_id=task_id,
            entry_type=EntryType.ANSWER_FORMAT,
            content=format_spec.get("format_specification", {}),
        )
        
        return [{"type": "answer_format", "content": format_spec.get("format_specification")}]


class SynthesizerAgent(BaseAgent):
    """
    Synthesizes all information into a final coherent answer.
    """
    
    async def _execute_impl(self, task_id: str) -> List[Dict[str, Any]]:
        """Execute synthesis."""
        # Read all necessary information
        summaries = await self._read_blackboard(task_id, EntryType.TASK_SUMMARY)
        plans = await self._read_blackboard(task_id, EntryType.PLAN)
        knowledge = await self._read_blackboard(task_id, EntryType.RETRIEVED_KNOWLEDGE)
        formats = await self._read_blackboard(task_id, EntryType.ANSWER_FORMAT)
        
        if not summaries:
            logger.warning("No task summary found")
            return []
        
        # Prepare context
        task_summary = summaries[-1].content
        plan_text = str(plans[-1].content) if plans else "計画なし"
        knowledge_text = str(knowledge[-1].content) if knowledge else "知識なし"
        format_text = str(formats[-1].content) if formats else "形式指定なし"
        
        context = {
            "task_summary": task_summary,
            "plan": plan_text,
            "retrieved_knowledge": knowledge_text,
            "answer_format": format_text,
        }
        
        messages = self._format_prompt(
            user_content=task_summary,
            context=context,
        )
        
        # Generate final answer
        response = await self._generate_response(messages)
        
        # Write final answer
        await self._write_blackboard(
            task_id=task_id,
            entry_type=EntryType.FINAL_ANSWER,
            content=response,
        )
        
        return [{"type": "final_answer", "content": response}]


class ConductorAgent(BaseAgent):
    """
    Monitors the blackboard and coordinates agent activities.
    """
    
    async def _execute_impl(self, task_id: str) -> List[Dict[str, Any]]:
        """Execute monitoring and coordination."""
        # Get current blackboard state
        state = await self.blackboard.get_state(task_id)
        if not state:
            return []
        
        # Format blackboard state
        blackboard_summary = await self.blackboard.get_task_summary(task_id)
        blackboard_state_text = f"""
タスクステータス: {blackboard_summary['status']}
総エントリー数: {blackboard_summary['total_entries']}
エントリータイプ別:
{chr(10).join(f"  - {k}: {v}" for k, v in blackboard_summary['entry_types'].items())}
エージェント活動:
{chr(10).join(f"  - {k}: {v}回" for k, v in blackboard_summary['agent_activity'].items())}
"""
        
        # Prepare prompt
        context = {"blackboard_state": blackboard_state_text}
        
        messages = self._format_prompt(
            user_content=blackboard_state_text,
            context=context,
        )
        
        # Generate assessment
        response = await self._generate_response(messages)
        
        # Parse JSON response
        assessment = self._parse_json_response(response)
        
        # Write conductor directive
        await self._write_blackboard(
            task_id=task_id,
            entry_type=EntryType.CONDUCTOR_DIRECTIVE,
            content=assessment,
        )
        
        return [{"type": "conductor_directive", "content": assessment}]
