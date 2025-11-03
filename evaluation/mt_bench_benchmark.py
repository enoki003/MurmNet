"""
MT-Bench style multi-turn conversation evaluation.
Tests dialogue coherence and instruction following.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from evaluation.base_benchmark import BaseBenchmark


class MTBenchBenchmark(BaseBenchmark):
    """
    MT-Bench style evaluation.
    
    Tests multi-turn conversation ability and instruction following.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_name = "MT-Bench"
    
    async def load_dataset(self) -> List[Dict[str, Any]]:
        """Load MT-Bench questions from downloaded file."""
        logger.info("Loading MT-Bench questions...")
        
        try:
            # Load from downloaded JSONL file
            data_file = Path("evaluation/benchmarks/mt_bench_questions.jsonl")
            
            if not data_file.exists():
                logger.error(f"MT-Bench data file not found: {data_file}")
                logger.info("Please run: python evaluation/download_benchmarks.py")
                return []
            
            questions = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    turns = item.get("turns", [])
                    questions.append({
                        "id": f"mt_bench_{item['question_id']}",
                        "category": item.get("category", "general"),
                        "question": turns[0] if len(turns) > 0 else "",
                        "second_turn": turns[1] if len(turns) > 1 else "",
                        "answer": None,  # No ground truth for MT-Bench
                    })
            
            logger.info(f"Loaded {len(questions)} MT-Bench questions")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to load MT-Bench questions: {e}")
            return []
    
    def evaluate_answer(
        self,
        question: Dict[str, Any],
        generated_answer: str,
    ) -> float:
        """
        Evaluate generated answer quality.
        
        For MT-Bench, we use heuristic scoring based on:
        - Answer length (not too short, not too long)
        - Presence of relevant keywords
        - Format appropriateness
        
        Args:
            question: Question dict
            generated_answer: Generated answer text
            
        Returns:
            Score (0.0 to 1.0)
        """
        if not generated_answer or len(generated_answer.strip()) < 10:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check length (reasonable range)
        length = len(generated_answer)
        if 50 <= length <= 500:
            score += 0.2
        
        # Check for Japanese content (since questions are in Japanese)
        japanese_chars = sum(1 for c in generated_answer if '\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff')
        if japanese_chars > length * 0.1:  # At least 10% Japanese
            score += 0.2
        
        # Category-specific checks
        category = question.get("category", "")
        
        if category == "math":
            # Check for numbers
            import re
            if re.search(r'\d+', generated_answer):
                score += 0.1
        
        elif category == "writing":
            # Check for poetic elements (line breaks, decent length)
            if '\n' in generated_answer and length > 30:
                score += 0.1
        
        elif category == "extraction":
            # Check if answer is concise
            if length < 200:
                score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
