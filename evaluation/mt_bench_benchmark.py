"""
MT-Bench style multi-turn conversation evaluation.
Tests dialogue coherence and instruction following.
"""

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
        
        # Predefined multi-turn questions
        self.questions = [
            {
                "category": "reasoning",
                "turns": [
                    "3つの箱があります。赤い箱には5個のりんご、青い箱には3個のりんご、緑の箱には2個のりんごが入っています。全部で何個のりんごがありますか？",
                    "では、赤い箱から2個、青い箱から1個のりんごを取り出しました。今、箱の中には全部で何個のりんごが残っていますか？",
                ],
            },
            {
                "category": "writing",
                "turns": [
                    "「春」をテーマにした短い詩を書いてください。",
                    "その詩をもとに、秋をテーマにしたバージョンも作ってください。",
                ],
            },
            {
                "category": "roleplay",
                "turns": [
                    "あなたは親切な図書館司書です。おすすめの本を3冊教えてください。",
                    "その中で、初心者に最も適している本はどれですか？その理由も説明してください。",
                ],
            },
            {
                "category": "extraction",
                "turns": [
                    "次の文から人物名を抽出してください: 「太郎と花子は公園で遊んでいました。そこに次郎が来ました。」",
                    "では、彼らがいた場所はどこですか？",
                ],
            },
            {
                "category": "math",
                "turns": [
                    "12 × 15 を計算してください。",
                    "その答えを3で割ると、いくつになりますか？",
                ],
            },
        ]
    
    async def load_dataset(self) -> List[Dict[str, Any]]:
        """Load MT-Bench questions."""
        logger.info("Loading MT-Bench questions...")
        
        questions = []
        for i, item in enumerate(self.questions):
            questions.append({
                "id": f"mt_bench_{i}",
                "category": item["category"],
                "question": item["turns"][0],  # First turn
                "second_turn": item["turns"][1],  # Second turn
                "answer": None,  # No ground truth for MT-Bench
            })
        
        logger.info(f"Loaded {len(questions)} MT-Bench questions")
        return questions
    
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
