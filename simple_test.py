#!/usr/bin/env python3
"""
Simple test to verify GPT-2 model loading and basic generation.
"""

from src.agents.llm import LanguageModel
from src.config import config

def test_gpt2():
    print("=" * 70)
    print("GPT-2 MODEL TEST")
    print("=" * 70)
    
    # Load configuration
    print(f"\nModel configured: {config.model.default_model_name}")
    
    # Initialize LLM
    print("\nInitializing Language Model...")
    llm = LanguageModel()
    
    # Test simple generation
    print("\nTesting simple generation...")
    prompt = "What is machine learning?"
    response = llm.generate(prompt, max_tokens=50)
    
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print("\n✓ GPT-2 model loaded and working!")

if __name__ == "__main__":
    test_gpt2()
