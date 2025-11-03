#!/usr/bin/env python
"""
Simple evaluation runner for testing the benchmark system.
Runs a small subset of each benchmark to verify functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from evaluation.gsm8k_benchmark import GSM8KBenchmark
from evaluation.mt_bench_benchmark import MTBenchBenchmark
from evaluation.truthfulqa_benchmark import TruthfulQABenchmark


async def test_benchmark_loading():
    """Test that all benchmarks can load their data."""
    
    print("=" * 70)
    print("BENCHMARK DATA LOADING TEST")
    print("=" * 70)
    
    results = {}
    
    # Test GSM8K
    print("\n[1/3] Testing GSM8K...")
    gsm8k = GSM8KBenchmark(
        orchestrator=None,  # Not needed for data loading test
        output_dir=Path("evaluation/results/test")
    )
    gsm8k_data = await gsm8k.load_dataset()
    results["GSM8K"] = len(gsm8k_data) > 0
    if gsm8k_data:
        print(f"  ✓ Loaded {len(gsm8k_data)} questions")
        print(f"  Example: {gsm8k_data[0]['question'][:80]}...")
    else:
        print("  ✗ Failed to load data")
    
    # Test TruthfulQA
    print("\n[2/3] Testing TruthfulQA...")
    truthfulqa = TruthfulQABenchmark(
        orchestrator=None,  # Not needed for data loading test
        output_dir=Path("evaluation/results/test")
    )
    truthfulqa_data = await truthfulqa.load_dataset()
    results["TruthfulQA"] = len(truthfulqa_data) > 0
    if truthfulqa_data:
        print(f"  ✓ Loaded {len(truthfulqa_data)} questions")
        print(f"  Example: {truthfulqa_data[0]['question'][:80]}...")
    else:
        print("  ✗ Failed to load data")
    
    # Test MT-Bench
    print("\n[3/3] Testing MT-Bench...")
    mtbench = MTBenchBenchmark(
        orchestrator=None,  # Not needed for data loading test
        output_dir=Path("evaluation/results/test")
    )
    mtbench_data = await mtbench.load_dataset()
    results["MT-Bench"] = len(mtbench_data) > 0
    if mtbench_data:
        print(f"  ✓ Loaded {len(mtbench_data)} questions")
        print(f"  Example: {mtbench_data[0]['question'][:80]}...")
    else:
        print("  ✗ Failed to load data")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for benchmark, success in results.items():
        status = "✓ READY" if success else "✗ FAILED"
        print(f"{benchmark:15s}: {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n✓ All benchmarks are ready for evaluation!")
        print("\nNext steps:")
        print("  1. Start the API server: python src/main.py")
        print("  2. Run full evaluation: python evaluation/run_evaluation.py")
        return 0
    else:
        print("\n✗ Some benchmarks failed to load")
        print("\nPlease run: python evaluation/download_benchmarks.py")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(test_benchmark_loading()))
