"""
GSM8K benchmark evaluation.
Tests multi-step reasoning and mathematical problem solving.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from evaluation.base_benchmark import BaseBenchmark


class GSM8KBenchmark(BaseBenchmark):
    """
    GSM8K (Grade School Math 8K) benchmark.
    
    Tests the model's ability to solve grade-school level math word problems
    that require multi-step reasoning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_name = "GSM8K"
    
    async def load_dataset(self) -> List[Dict[str, Any]]:
        """Load GSM8K dataset from downloaded file."""
        logger.info("Loading GSM8K dataset...")
        
        try:
            # Load from downloaded JSONL file
            data_file = Path("evaluation/benchmarks/gsm8k_test.jsonl")
            
            if not data_file.exists():
                logger.error(f"GSM8K data file not found: {data_file}")
                logger.info("Please run: python evaluation/download_benchmarks.py")
                return []
            
            questions = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    item = json.loads(line)
                    questions.append({
                        "id": f"gsm8k_{i}",
                        "question": item["question"],
                        "answer": self._extract_answer(item["answer"]),
                        "full_solution": item["answer"],
                    })
            
            logger.info(f"Loaded {len(questions)} GSM8K questions")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to load GSM8K dataset: {e}")
            return []
    
    def _extract_answer(self, solution_text: str) -> str:
        """
        Extract the final numerical answer from GSM8K solution.
        
        Args:
            solution_text: Full solution text
            
        Returns:
            Numerical answer as string
        """
        # GSM8K answers are in format: "#### 123"
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', solution_text)
        if match:
            # Remove commas from number
            return match.group(1).replace(',', '')
        return ""
    
    def evaluate_answer(
        self,
        question: Dict[str, Any],
        generated_answer: str,
    ) -> float:
        """
        Evaluate generated answer against expected answer.
        
        Args:
            question: Question dict
            generated_answer: Generated answer text
            
        Returns:
            Score (1.0 if correct, 0.0 if incorrect)
        """
        expected_answer = question["answer"]
        
        # Extract numerical answer from generated text
        # Look for numbers in the generated answer
        numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', generated_answer)
        
        if not numbers:
            return 0.0
        
        # Check if any extracted number matches the expected answer
        for num_str in numbers:
            num_str = num_str.replace(',', '')
            
            try:
                # Try to compare as floats
                generated_num = float(num_str)
                expected_num = float(expected_answer)
                
                # Allow small floating point differences
                if abs(generated_num - expected_num) < 0.01:
                    return 1.0
            except ValueError:
                # If conversion fails, try string comparison
                if num_str == expected_answer:
                    return 1.0
        
        return 0.0
