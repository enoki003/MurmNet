#!/usr/bin/env python3
"""
Run evaluation benchmarks on MurmurNet system.
Tests with GSM8K, TruthfulQA, and MT-Bench.
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

from src.orchestrator.orchestrator import Orchestrator
from evaluation.gsm8k_benchmark import GSM8KBenchmark
from evaluation.truthfulqa_benchmark import TruthfulQABenchmark
from evaluation.mt_bench_benchmark import MTBenchBenchmark


async def run_gsm8k_evaluation(
    orchestrator: Orchestrator,
    output_dir: Path,
    num_samples: int = 1,
):
    """Run GSM8K evaluation."""
    print("=" * 70)
    print("GSM8K EVALUATION")
    print("=" * 70)
    
    benchmark = GSM8KBenchmark(orchestrator=orchestrator, output_dir=output_dir)
    summary = await benchmark.run_evaluation(max_questions=num_samples)
    
    total = summary.get("total_questions", 0)
    correct = summary.get("perfect_score_count", 0)
    accuracy = summary.get("average_score", 0.0)
    avg_time = summary.get("average_execution_time_seconds", 0.0)
    
    print(f"\nGSM8K Results:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Average Time: {avg_time:.2f}s")
    
    return summary


async def run_truthfulqa_evaluation(
    orchestrator: Orchestrator,
    output_dir: Path,
    num_samples: int = 1,
):
    """Run TruthfulQA evaluation."""
    print("\n" + "=" * 70)
    print("TRUTHFULQA EVALUATION")
    print("=" * 70)
    
    benchmark = TruthfulQABenchmark(orchestrator=orchestrator, output_dir=output_dir)
    summary = await benchmark.run_evaluation(max_questions=num_samples)
    
    total = summary.get("total_questions", 0)
    truthful = summary.get("perfect_score_count", 0)
    truthfulness = summary.get("average_score", 0.0)
    avg_time = summary.get("average_execution_time_seconds", 0.0)
    
    print(f"\nTruthfulQA Results:")
    print(f"  Total: {total}")
    print(f"  Truthful: {truthful}")
    print(f"  Truthfulness: {truthfulness:.2%}")
    print(f"  Average Time: {avg_time:.2f}s")
    
    return summary


async def run_mtbench_evaluation(
    orchestrator: Orchestrator,
    output_dir: Path,
    max_questions: int = 2,
):
    """Run MT-Bench evaluation."""
    print("\n" + "=" * 70)
    print("MT-BENCH EVALUATION")
    print("=" * 70)
    
    benchmark = MTBenchBenchmark(orchestrator=orchestrator, output_dir=output_dir)
    summary = await benchmark.run_evaluation(max_questions=max_questions)
    
    total = summary.get("total_questions", 0)
    avg_score = summary.get("average_score", 0.0) * 10  # scale to 10
    avg_time = summary.get("average_execution_time_seconds", 0.0)
    category_scores = summary.get("category_scores", {})
    
    print(f"\nMT-Bench Results:")
    print(f"  Total Questions: {total}")
    print(f"  Average Score: {avg_score:.2f}/10")
    print(f"  Average Time: {avg_time:.2f}s")
    
    if category_scores:
        print("\n  Category Scores:")
        for category, score in category_scores.items():
            print(f"    {category}: {score * 10:.2f}/10")
    
    summary["average_score_scaled_10"] = avg_score
    summary["category_scores_scaled_10"] = {
        category: value * 10 for category, value in category_scores.items()
    }
    
    return summary


async def main():
    """Run all evaluations."""
    print("=" * 70)
    print("MURMURNET EVALUATION")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results_root = Path("evaluation/results")
    results_root.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator
    print("Initializing orchestrator...")
    orchestrator = Orchestrator()
    print("✓ Orchestrator initialized\n")
    
    # Run evaluations
    all_results = {}
    
    try:
        # GSM8K (50 samples for speed)
        gsm8k_results = await run_gsm8k_evaluation(
            orchestrator,
            results_root / "gsm8k",
            num_samples=1,
        )
        all_results['gsm8k'] = gsm8k_results
        
        # TruthfulQA (50 samples)
        truthfulqa_results = await run_truthfulqa_evaluation(
            orchestrator,
            results_root / "truthfulqa",
            num_samples=1,
        )
        all_results['truthfulqa'] = truthfulqa_results
        
        # MT-Bench (all 8 questions)
        mtbench_results = await run_mtbench_evaluation(
            orchestrator,
            results_root / "mt_bench",
            max_questions=2,
        )
        all_results['mt_bench'] = mtbench_results
        
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_file}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
