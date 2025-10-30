"""
Main evaluation script.
Runs all benchmarks and generates comparison report.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List

from loguru import logger

from src.config import config
from src.orchestrator import Orchestrator
from src.knowledge import RAGSystem, ZIMParser
from src.memory import LongTermMemory, ExperienceMemory
from src.utils.logging import setup_logging

from evaluation.gsm8k_benchmark import GSM8KBenchmark
from evaluation.mt_bench_benchmark import MTBenchBenchmark
from evaluation.truthfulqa_benchmark import TruthfulQABenchmark


async def run_all_benchmarks(
    orchestrator: Orchestrator,
    output_dir: Path,
    max_questions_per_benchmark: int = 100,
) -> Dict[str, any]:
    """
    Run all benchmark evaluations.
    
    Args:
        orchestrator: Orchestrator instance
        output_dir: Output directory for results
        max_questions_per_benchmark: Max questions per benchmark
        
    Returns:
        Combined results dictionary
    """
    results = {}
    
    # GSM8K
    logger.info("=" * 60)
    logger.info("Running GSM8K Benchmark")
    logger.info("=" * 60)
    gsm8k = GSM8KBenchmark(orchestrator, output_dir)
    results["gsm8k"] = await gsm8k.run_evaluation(
        max_questions=max_questions_per_benchmark
    )
    
    # MT-Bench
    logger.info("=" * 60)
    logger.info("Running MT-Bench Benchmark")
    logger.info("=" * 60)
    mt_bench = MTBenchBenchmark(orchestrator, output_dir)
    results["mt_bench"] = await mt_bench.run_evaluation()
    
    # TruthfulQA
    logger.info("=" * 60)
    logger.info("Running TruthfulQA Benchmark")
    logger.info("=" * 60)
    truthfulqa = TruthfulQABenchmark(orchestrator, output_dir)
    results["truthfulqa"] = await truthfulqa.run_evaluation(
        max_questions=max_questions_per_benchmark
    )
    
    return results


def generate_comparison_report(
    results: Dict[str, any],
    output_dir: Path,
) -> None:
    """
    Generate comparison report across benchmarks.
    
    Args:
        results: Combined benchmark results
        output_dir: Output directory
    """
    report_lines = [
        "=" * 80,
        "MurmurNet Evaluation Report",
        "=" * 80,
        "",
    ]
    
    for benchmark_name, summary in results.items():
        report_lines.append(f"{benchmark_name.upper()}:")
        report_lines.append(f"  Total Questions: {summary.get('total_questions', 0)}")
        report_lines.append(f"  Average Score: {summary.get('average_score', 0):.3f}")
        report_lines.append(f"  Perfect Scores: {summary.get('perfect_score_count', 0)}")
        report_lines.append(f"  Failures: {summary.get('failure_count', 0)}")
        report_lines.append(f"  Avg Execution Time: {summary.get('average_execution_time_seconds', 0):.2f}s")
        report_lines.append("")
    
    # Overall statistics
    total_questions = sum(s.get('total_questions', 0) for s in results.values())
    if total_questions > 0:
        overall_avg = sum(
            s.get('average_score', 0) * s.get('total_questions', 0)
            for s in results.values()
        ) / total_questions
        
        report_lines.append("OVERALL:")
        report_lines.append(f"  Total Questions Across All Benchmarks: {total_questions}")
        report_lines.append(f"  Weighted Average Score: {overall_avg:.3f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    report_file = output_dir / "evaluation_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Also save as JSON
    json_file = output_dir / "evaluation_summary.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print report
    print(report_text)
    
    logger.info(f"Report saved to {report_file}")
    logger.info(f"Summary saved to {json_file}")


async def main():
    """Main evaluation entry point."""
    # Setup logging
    setup_logging()
    
    logger.info("Starting MurmurNet Evaluation")
    
    # Ensure directories exist
    config.ensure_directories()
    
    # Initialize RAG system (optional)
    rag_system = None
    if config.knowledge_base.zim_file_path and config.knowledge_base.zim_file_path.exists():
        try:
            logger.info("Initializing RAG system...")
            zim_parser = ZIMParser()
            rag_system = RAGSystem(zim_parser=zim_parser)
            rag_system.load_database()
            logger.info("RAG system initialized")
        except Exception as e:
            logger.warning(f"RAG system initialization failed: {e}")
    
    # Initialize memory systems
    long_term_memory = LongTermMemory()
    experience_memory = ExperienceMemory()
    
    # Initialize orchestrator
    orchestrator = Orchestrator(
        rag_system=rag_system,
        long_term_memory=long_term_memory,
        experience_memory=experience_memory,
    )
    
    # Run benchmarks
    output_dir = config.evaluation.evaluation_output_dir
    results = await run_all_benchmarks(
        orchestrator=orchestrator,
        output_dir=output_dir,
        max_questions_per_benchmark=50,  # Adjust as needed
    )
    
    # Generate report
    generate_comparison_report(results, output_dir)
    
    # Cleanup
    orchestrator.llm.unload()
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
