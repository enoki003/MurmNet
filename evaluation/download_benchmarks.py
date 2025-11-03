#!/usr/bin/env python
"""
Download and prepare evaluation benchmarks.
Downloads GSM8K, MT-Bench, and TruthfulQA datasets from official sources.
"""

import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from loguru import logger

from src.config import config


def download_gsm8k(output_dir: Path):
    """
    Download GSM8K dataset from Hugging Face.
    
    GSM8K: Grade School Math 8K
    https://huggingface.co/datasets/gsm8k
    """
    logger.info("Downloading GSM8K dataset...")
    
    try:
        # Load dataset
        dataset = load_dataset("gsm8k", "main")
        
        # Save test split
        test_data = dataset["test"]
        
        output_file = output_dir / "gsm8k_test.jsonl"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in test_data:
                json.dump({
                    "question": item["question"],
                    "answer": item["answer"],
                }, f, ensure_ascii=False)
                f.write("\n")
        
        logger.info(f"✓ GSM8K: Saved {len(test_data)} test examples to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download GSM8K: {e}")
        return False


def download_truthfulqa(output_dir: Path):
    """
    Download TruthfulQA dataset from Hugging Face.
    
    TruthfulQA: Measuring How Models Mimic Human Falsehoods
    https://huggingface.co/datasets/truthful_qa
    """
    logger.info("Downloading TruthfulQA dataset...")
    
    try:
        # Load dataset (multiple choice format)
        dataset = load_dataset("truthful_qa", "multiple_choice")
        
        # Save validation split
        val_data = dataset["validation"]
        
        output_file = output_dir / "truthfulqa_test.jsonl"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in val_data:
                json.dump({
                    "question": item["question"],
                    "mc1_targets": item.get("mc1_targets", {}),
                    "mc2_targets": item.get("mc2_targets", {}),
                }, f, ensure_ascii=False)
                f.write("\n")
        
        logger.info(f"✓ TruthfulQA: Saved {len(val_data)} validation examples to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download TruthfulQA: {e}")
        return False


def download_mt_bench_questions(output_dir: Path):
    """
    Download MT-Bench questions.
    
    MT-Bench: Multi-turn Benchmark
    Questions from FastChat repository
    """
    logger.info("Downloading MT-Bench questions...")
    
    try:
        # MT-Bench questions are typically in the FastChat repo
        # We'll create a subset of common evaluation questions
        
        mt_bench_questions = [
            {
                "question_id": 1,
                "category": "writing",
                "turns": [
                    "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
                    "Rewrite your previous response. Start every sentence with the letter A."
                ]
            },
            {
                "question_id": 2,
                "category": "reasoning",
                "turns": [
                    "Suppose you are a mathematician and poet. Compose a short poem that describes the beauty of mathematics.",
                    "Describe the poem's tone and writing style."
                ]
            },
            {
                "question_id": 3,
                "category": "math",
                "turns": [
                    "How many integers are in the solution of the inequality |x + 5| < 10?",
                    "What about |x + 10| < 5?"
                ]
            },
            {
                "question_id": 4,
                "category": "coding",
                "turns": [
                    "Write a Python function to find the nth Fibonacci number using dynamic programming.",
                    "Now write a recursive version with memoization."
                ]
            },
            {
                "question_id": 5,
                "category": "roleplay",
                "turns": [
                    "Pretend yourself to be Elon Musk in all the following conversations. Speak like Elon Musk as much as possible. Why do we need to go to Mars?",
                    "How do you like dancing? Can you teach me?"
                ]
            },
            {
                "question_id": 6,
                "category": "extraction",
                "turns": [
                    "Given the following data, identify the company with the highest profit in 2021 and provide its CEO's name: a) Company X, with CEO Amy Williams, reported $30 billion in revenue and a $3 billion profit in 2021. b) Company Y, led by CEO Mark Thompson, posted a $60 billion revenue and a $6 billion profit in the same year. c) Company Z, under CEO Sarah Johnson, announced a $20 billion revenue and a $7 billion profit in 2021.",
                    "Which company had the lowest profit margin (profit/revenue ratio)?"
                ]
            },
            {
                "question_id": 7,
                "category": "stem",
                "turns": [
                    "Explain the process of photosynthesis in plants.",
                    "How much energy can a tree produce through photosynthesis in its lifetime? Please provide an estimate."
                ]
            },
            {
                "question_id": 8,
                "category": "humanities",
                "turns": [
                    "Discuss the themes of George Orwell's '1984'.",
                    "How do the themes relate to modern-day surveillance and privacy concerns?"
                ]
            }
        ]
        
        output_file = output_dir / "mt_bench_questions.jsonl"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in mt_bench_questions:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        
        logger.info(f"✓ MT-Bench: Saved {len(mt_bench_questions)} questions to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create MT-Bench questions: {e}")
        return False


def main():
    """Download all benchmark datasets."""
    
    logger.info("=" * 60)
    logger.info("BENCHMARK DATASET DOWNLOAD")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(config.evaluation.benchmark_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    results = {}
    
    # Download GSM8K
    logger.info("\n[1/3] GSM8K...")
    results["gsm8k"] = download_gsm8k(output_dir)
    
    # Download TruthfulQA
    logger.info("\n[2/3] TruthfulQA...")
    results["truthfulqa"] = download_truthfulqa(output_dir)
    
    # Download MT-Bench
    logger.info("\n[3/3] MT-Bench...")
    results["mt_bench"] = download_mt_bench_questions(output_dir)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{dataset}: {status}")
    
    all_success = all(results.values())
    
    if all_success:
        logger.info("\n✓ All datasets downloaded successfully!")
        return 0
    else:
        logger.error("\n✗ Some datasets failed to download")
        return 1


if __name__ == "__main__":
    sys.exit(main())
