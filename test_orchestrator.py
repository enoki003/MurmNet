#!/usr/bin/env python
"""
Simple test script for MurmurNet system without API server.
Tests orchestrator directly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from src.orchestrator.orchestrator import Orchestrator


async def test_simple_query():
    """Test a simple query through the orchestrator."""
    
    print("=" * 70)
    print("ORCHESTRATOR TEST")
    print("=" * 70)
    print()
    
    # Initialize orchestrator
    print("Initializing orchestrator...")
    orchestrator = Orchestrator()
    print("✓ Orchestrator initialized\n")
    
    # Test query
    query = "What is 2 + 2?"
    print(f"Query: {query}")
    print("Processing...\n")
    
    try:
        result = await orchestrator.process_query(query)
        
        print("=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"Task ID: {result.get('task_id', 'N/A')}")
        print(f"Success: {result.get('success', False)}")
        
        if result.get('answer'):
            print(f"\nAnswer:\n{result['answer']}")
        
        if result.get('error'):
            print(f"\nError: {result['error']}")
        
        # Show execution history
        if result.get('execution_history'):
            print(f"\n--- Execution History ({len(result['execution_history'])} phases) ---")
            for i, phase in enumerate(result['execution_history'], 1):
                print(f"\n{i}. {phase.get('phase', 'Unknown')}")
                print(f"   Agents: {', '.join(phase.get('agents', []))}")
                if phase.get('error'):
                    print(f"   Error: {phase['error']}")
        
        return 0 if result.get('success') else 1
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        logger.exception("Test failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(test_simple_query()))
