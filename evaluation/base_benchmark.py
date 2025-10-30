"""
Base evaluation framework for benchmarking the system.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from tqdm import tqdm

from src.orchestrator import Orchestrator


class EvaluationResult(dict):
    """Result of a single evaluation."""
    
    def __init__(
        self,
        question_id: str,
        question: str,
        expected_answer: Optional[str],
        generated_answer: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            question_id=question_id,
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            score=score,
            metadata=metadata or {},
        )


class BaseBenchmark(ABC):
    """
    Base class for benchmark evaluations.
    """
    
    def __init__(
        self,
        orchestrator: Orchestrator,
        output_dir: Path,
    ):
        """
        Initialize benchmark.
        
        Args:
            orchestrator: Orchestrator instance
            output_dir: Directory to save results
        """
        self.orchestrator = orchestrator
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_name = self.__class__.__name__
        
    @abstractmethod
    async def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load benchmark dataset.
        
        Returns:
            List of question dicts
        """
        pass
    
    @abstractmethod
    def evaluate_answer(
        self,
        question: Dict[str, Any],
        generated_answer: str,
    ) -> float:
        """
        Evaluate a generated answer.
        
        Args:
            question: Question dict
            generated_answer: Generated answer
            
        Returns:
            Score (0.0 to 1.0)
        """
        pass
    
    async def run_evaluation(
        self,
        max_questions: Optional[int] = None,
        save_intermediate: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete benchmark evaluation.
        
        Args:
            max_questions: Maximum number of questions to evaluate
            save_intermediate: Save results after each question
            
        Returns:
            Evaluation summary
        """
        logger.info(f"Starting {self.benchmark_name} evaluation")
        
        # Load dataset
        dataset = await self.load_dataset()
        
        if max_questions:
            dataset = dataset[:max_questions]
        
        logger.info(f"Loaded {len(dataset)} questions")
        
        # Run evaluation
        results = []
        total_score = 0.0
        
        for i, question in enumerate(tqdm(dataset, desc=f"{self.benchmark_name}")):
            try:
                # Generate answer
                response = await self.orchestrator.process_query(question["question"])
                
                if not response["success"]:
                    logger.warning(f"Question {i+1} failed: {response.get('error')}")
                    score = 0.0
                    generated_answer = f"Error: {response.get('error')}"
                else:
                    generated_answer = response["answer"]
                    score = self.evaluate_answer(question, generated_answer)
                
                # Create result
                result = EvaluationResult(
                    question_id=question.get("id", f"q{i+1}"),
                    question=question["question"],
                    expected_answer=question.get("answer"),
                    generated_answer=generated_answer,
                    score=score,
                    metadata={
                        "execution_time": response.get("execution_time_seconds"),
                        "task_id": response.get("task_id"),
                    },
                )
                
                results.append(result)
                total_score += score
                
                # Save intermediate results
                if save_intermediate and (i + 1) % 10 == 0:
                    self._save_intermediate_results(results, i + 1)
                
            except Exception as e:
                logger.error(f"Error evaluating question {i+1}: {e}")
                result = EvaluationResult(
                    question_id=question.get("id", f"q{i+1}"),
                    question=question["question"],
                    expected_answer=question.get("answer"),
                    generated_answer=f"Error: {str(e)}",
                    score=0.0,
                )
                results.append(result)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Save final results
        self._save_final_results(results, summary)
        
        logger.info(
            f"{self.benchmark_name} evaluation complete. "
            f"Average score: {summary['average_score']:.3f}"
        )
        
        return summary
    
    def _calculate_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not results:
            return {}
        
        total_score = sum(r["score"] for r in results)
        average_score = total_score / len(results)
        
        # Count perfect scores
        perfect_count = sum(1 for r in results if r["score"] == 1.0)
        
        # Count failures
        failure_count = sum(1 for r in results if r["score"] == 0.0)
        
        # Average execution time
        exec_times = [
            r["metadata"].get("execution_time", 0)
            for r in results
            if r["metadata"].get("execution_time")
        ]
        avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
        
        return {
            "benchmark_name": self.benchmark_name,
            "total_questions": len(results),
            "average_score": average_score,
            "perfect_score_count": perfect_count,
            "failure_count": failure_count,
            "average_execution_time_seconds": avg_exec_time,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def _save_intermediate_results(
        self,
        results: List[EvaluationResult],
        count: int,
    ) -> None:
        """Save intermediate results."""
        filepath = self.output_dir / f"{self.benchmark_name}_intermediate_{count}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved intermediate results to {filepath}")
    
    def _save_final_results(
        self,
        results: List[EvaluationResult],
        summary: Dict[str, Any],
    ) -> None:
        """Save final results and summary."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"{self.benchmark_name}_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(
                {"summary": summary, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )
        
        # Save summary only
        summary_file = self.output_dir / f"{self.benchmark_name}_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        logger.info(f"Saved summary to {summary_file}")
