#!/usr/bin/env python
"""Test LLM agents with a small model"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.agents.llm import LanguageModel
from src.agents.specialized_agents import (
    InputAnalyzerAgent,
    PlannerAgent,
    ResponseFormatterAgent,
    SynthesizerAgent,
)
from src.blackboard.blackboard import BlackboardSystem
from src.blackboard.models import EntryType
from loguru import logger


async def test_llm_engine():
    """Test LLM engine with small model"""
    print("\n=== Testing LLM Engine ===")
    
    llm = LanguageModel()
    
    # Test simple generation
    response = await llm.generate(
        prompt="Translate 'Hello' to French:",
        max_new_tokens=10
    )
    print(f"✓ LLM response: {response[:100]}...")
    print("✓ LLM Engine test passed\n")


async def test_input_analyzer():
    """Test Input Analyzer agent"""
    print("=== Testing Input Analyzer Agent ===")
    
    bb = BlackboardSystem()
    state = await bb.create_task()
    task_id = state.task_id
    
    # Write user input
    await bb.write_entry(
        task_id=task_id,
        agent_id="user",
        entry_type=EntryType.USER_INPUT,
        content="What is machine learning?"
    )
    
    # Create and execute agent
    agent = InputAnalyzerAgent()
    result = await agent.execute(task_id)
    
    print(f"✓ Agent executed successfully")
    print(f"  Output: {result.get('output', '')[:100]}...")
    print("✓ Input Analyzer test passed\n")


async def test_planner():
    """Test Planner agent"""
    print("=== Testing Planner Agent ===")
    
    bb = BlackboardSystem()
    state = await bb.create_task()
    task_id = state.task_id
    
    # Write necessary entries
    await bb.write_entry(
        task_id=task_id,
        agent_id="user",
        entry_type=EntryType.USER_INPUT,
        content="Explain neural networks"
    )
    await bb.write_entry(
        task_id=task_id,
        agent_id="analyzer",
        entry_type=EntryType.TASK_SUMMARY,
        content="User wants explanation of neural networks"
    )
    
    # Create and execute agent
    agent = PlannerAgent()
    result = await agent.execute(task_id)
    
    print(f"✓ Agent executed successfully")
    print(f"  Output: {result.get('output', '')[:100]}...")
    print("✓ Planner test passed\n")


async def test_synthesizer():
    """Test Synthesizer agent"""
    print("=== Testing Synthesizer Agent ===")
    
    bb = BlackboardSystem()
    state = await bb.create_task()
    task_id = state.task_id
    
    # Write necessary entries
    await bb.write_entry(
        task_id=task_id,
        agent_id="user",
        entry_type=EntryType.USER_INPUT,
        content="What is Python?"
    )
    await bb.write_entry(
        task_id=task_id,
        agent_id="knowledge",
        entry_type=EntryType.RETRIEVED_KNOWLEDGE,
        content="Python is a high-level programming language."
    )
    
    # Create and execute agent
    agent = SynthesizerAgent()
    result = await agent.execute(task_id)
    
    print(f"✓ Agent executed successfully")
    print(f"  Output: {result.get('output', '')[:100]}...")
    print("✓ Synthesizer test passed\n")


async def test_response_formatter():
    """Test Response Formatter agent"""
    print("=== Testing Response Formatter Agent ===")
    
    bb = BlackboardSystem()
    state = await bb.create_task()
    task_id = state.task_id
    
    # Write necessary entries
    await bb.write_entry(
        task_id=task_id,
        agent_id="user",
        entry_type=EntryType.USER_INPUT,
        content="Tell me about AI"
    )
    await bb.write_entry(
        task_id=task_id,
        agent_id="synthesizer",
        entry_type=EntryType.SYNTHESIZED_DRAFT,
        content="AI is artificial intelligence, which enables machines to learn and solve problems."
    )
    
    # Create and execute agent
    agent = ResponseFormatterAgent()
    result = await agent.execute(task_id)
    
    print(f"✓ Agent executed successfully")
    print(f"  Output: {result.get('output', '')[:100]}...")
    print("✓ Response Formatter test passed\n")


async def main():
    """Run all agent tests"""
    print("=" * 60)
    print("LLM AGENT TESTS")
    print("=" * 60)
    print(f"Model: {config.model.default_model_name}")
    print(f"Device: {config.model.model_device}")
    print()
    
    try:
        await test_llm_engine()
        await test_input_analyzer()
        await test_planner()
        await test_synthesizer()
        await test_response_formatter()
        
        print("=" * 60)
        print("✓ ALL AGENT TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        logger.exception("Agent test failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
