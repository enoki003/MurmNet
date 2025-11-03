#!/usr/bin/env python
"""Test basic components: Blackboard, Memory, Embeddings"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.blackboard.models import BlackboardEntry, EntryType
from src.blackboard.blackboard import BlackboardSystem
from src.knowledge.embeddings import EmbeddingModel
from src.memory.long_term_memory import LongTermMemory
from loguru import logger

async def test_blackboard():
    """Test Blackboard system"""
    print("\n=== Testing Blackboard ===")
    
    bb = BlackboardSystem()
    
    # Create a task
    state = await bb.create_task()
    task_id = state.task_id
    print(f"✓ Created task: {task_id}")
    
    # Add entries
    entry1 = await bb.write_entry(
        task_id=task_id,
        agent_id="test_agent",
        entry_type=EntryType.USER_INPUT,
        content="Test query"
    )
    print(f"✓ Added entry: {entry1.entry_id}")
    
    entry2 = await bb.write_entry(
        task_id=task_id,
        agent_id="test_agent",
        entry_type=EntryType.TASK_SUMMARY,
        content="Test analysis"
    )
    print(f"✓ Added entry: {entry2.entry_id}")
    
    # Retrieve entries
    current_state = await bb.get_state(task_id)
    entries = current_state.get_entries_by_type(EntryType.USER_INPUT)
    print(f"✓ Retrieved {len(entries)} USER_INPUT entries")
    
    # Get current state
    print(f"✓ Current state has {len(current_state.entries)} total entries")
    
    print("✓ Blackboard test passed\n")


async def test_embeddings():
    """Test Embedding model"""
    print("=== Testing Embeddings ===")
    
    embedding_model = EmbeddingModel()
    
    # Test single embedding
    text = "This is a test sentence for embedding."
    embedding = embedding_model.encode_single(text)
    print(f"✓ Single embedding shape: {len(embedding)}")
    print(f"  Expected dimension: {config.vector_db.vector_db_dimension}")
    
    # Test batch embedding
    texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence"
    ]
    embeddings = embedding_model.encode(texts)
    print(f"✓ Batch embeddings shape: {len(embeddings)}x{len(embeddings[0])}")
    
    print("✓ Embeddings test passed\n")


async def test_memory():
    """Test Long-term memory"""
    print("=== Testing Long-term Memory ===")
    
    # Initialize memory
    memory = LongTermMemory()
    
    # Add memories
    memory.store_memory("Python is a programming language", {"category": "fact"})
    memory.store_memory("Machine learning uses data to train models", {"category": "fact"})
    memory.store_memory("Neural networks are inspired by the brain", {"category": "fact"})
    print("✓ Stored 3 memories")
    
    # Retrieve similar memories
    results = memory.retrieve_memories("What is Python?", top_k=2)
    print(f"✓ Retrieved {len(results)} relevant memories:")
    for mem in results:
        print(f"  - Importance: {mem.importance_score:.2f}, Text: {mem.content[:50]}...")
    
    print("✓ Memory test passed\n")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("COMPONENT TESTS")
    print("=" * 60)
    
    try:
        await test_blackboard()
        await test_embeddings()
        await test_memory()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        logger.exception("Test failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
