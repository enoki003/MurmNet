"""
Base agent implementation.
Provides common functionality for all specialized agents.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.agents.llm import LanguageModel
from src.blackboard import BlackboardSystem, EntryType, EntryMetadata
from src.config import config
from src.config.agent_config import AgentDefinition


class AgentExecutionError(Exception):
    """Raised when agent execution fails."""
    pass


class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    
    Provides common functionality:
    - Access to blackboard
    - LLM interaction
    - Error handling and retries
    - Logging
    """
    
    def __init__(
        self,
        agent_definition: AgentDefinition,
        blackboard: BlackboardSystem,
        llm: Optional[LanguageModel] = None,
    ):
        """
        Initialize base agent.
        
        Args:
            agent_definition: Agent configuration
            blackboard: Blackboard system instance
            llm: Language model instance (shared across agents)
        """
        self.definition = agent_definition
        self.blackboard = blackboard
        self.llm = llm or LanguageModel()
        
        # Configuration
        self.agent_id = agent_definition.agent_id
        self.agent_name = agent_definition.agent_name
        self.max_retries = config.agent.max_agent_retries
        self.timeout = config.agent.agent_timeout_seconds
        
        logger.info(f"Initialized agent: {self.agent_name} ({self.agent_id})")
    
    async def can_execute(self, task_id: str) -> bool:
        """
        Check if this agent can execute based on blackboard state.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if agent can execute
        """
        state = await self.blackboard.get_state(task_id)
        if not state:
            return False
        
        # Check if any trigger entry types are present
        for trigger_type in self.definition.triggers:
            if trigger_type == "*":  # Conductor agent monitors all
                return len(state.entries) > 0
            
            try:
                entry_type = EntryType(trigger_type)
                if state.get_latest_entry_by_type(entry_type):
                    return True
            except ValueError:
                continue
        
        return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(AgentExecutionError),
    )
    async def execute(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Execute the agent's task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of entries created by this agent
            
        Raises:
            AgentExecutionError: If execution fails
        """
        logger.info(f"Agent {self.agent_name} starting execution for task {task_id}")
        
        start_time = datetime.utcnow()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_impl(task_id),
                timeout=self.timeout,
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"Agent {self.agent_name} completed in {execution_time:.2f}s"
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Agent {self.agent_name} timed out after {self.timeout}s")
            raise AgentExecutionError(f"Agent {self.agent_name} timed out")
        
        except Exception as e:
            logger.error(f"Agent {self.agent_name} failed: {e}")
            raise AgentExecutionError(f"Agent {self.agent_name} failed: {e}")
    
    @abstractmethod
    async def _execute_impl(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Implementation of agent's execution logic.
        Must be implemented by subclasses.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of created entry data
        """
        pass
    
    async def _read_blackboard(
        self,
        task_id: str,
        entry_type: Optional[EntryType] = None,
    ) -> List[Any]:
        """
        Read entries from blackboard.
        
        Args:
            task_id: Task identifier
            entry_type: Optional filter by entry type
            
        Returns:
            List of entries
        """
        return await self.blackboard.read_entries(
            task_id=task_id,
            entry_type=entry_type,
        )
    
    async def _write_blackboard(
        self,
        task_id: str,
        entry_type: EntryType,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write entry to blackboard.
        
        Args:
            task_id: Task identifier
            entry_type: Type of entry
            content: Content to write
            metadata: Optional metadata
        """
        entry_metadata = EntryMetadata()
        if metadata:
            entry_metadata.additional_info = metadata
        
        await self.blackboard.write_entry(
            task_id=task_id,
            agent_id=self.agent_id,
            entry_type=entry_type,
            content=content,
            metadata=entry_metadata,
        )
        
        logger.debug(
            f"Agent {self.agent_name} wrote {entry_type.value} to blackboard"
        )
    
    def _format_prompt(
        self,
        user_content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Format prompt for LLM using agent's template.
        
        Args:
            user_content: User/task-specific content
            context: Additional context for template variables
            
        Returns:
            List of message dicts
        """
        # Fill in template variables
        user_prompt = self.definition.prompt_template.user_prompt_template
        
        if context:
            for key, value in context.items():
                placeholder = f"{{{key}}}"
                if placeholder in user_prompt:
                    user_prompt = user_prompt.replace(placeholder, str(value))
        
        # Replace main user content
        user_prompt = user_prompt.replace("{user_input}", user_content)
        
        messages = [
            {"role": "system", "content": self.definition.prompt_template.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return messages
    
    async def _generate_response(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Generated text
        """
        try:
            response = self.llm.generate_with_messages(
                messages=messages,
                max_tokens=self.definition.max_tokens,
                temperature=self.definition.temperature,
                top_p=self.definition.top_p,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise AgentExecutionError(f"LLM generation failed: {e}")
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON dict
            
        Raises:
            AgentExecutionError: If parsing fails
        """
        try:
            # Try to find JSON in the response
            # Look for content between first { and last }
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx + 1]
            parsed = json.loads(json_str)
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from response: {response[:200]}...")
            raise AgentExecutionError(f"JSON parsing failed: {e}")
