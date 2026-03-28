# FILE: src/evaluation/evaluator.py
"""
PubMedQA Evaluator for MIRAGE Benchmark.

Runs the MedicalAgent against PubMedQA questions and calculates accuracy.
Handles answer extraction from free-form LLM responses.

Example Usage:
    >>> from src.agent import MedicalAgent
    >>> from src.evaluation import PubMedQADataset, PubMedQAEvaluator
    >>> 
    >>> agent = MedicalAgent()
    >>> dataset = PubMedQADataset()
    >>> evaluator = PubMedQAEvaluator(agent, dataset)
    >>> 
    >>> results = await evaluator.evaluate(limit=50)
    >>> print(results.summary())

ENV_VARS:
    EVAL_DELAY_SECONDS: Delay between questions to respect rate limits (default: 1.0)
    EVAL_SAVE_INTERMEDIATE: Save results after each question (default: true)
"""

import asyncio
import json
import logging
import os
import re
import time
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.pubmedqa_dataset import PubMedQADataset, PubMedQAQuestion
from src.exceptions import TransientError

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single question.
    
    Attributes:
        question_id: Unique identifier for the question.
        question: The question text.
        correct_answer: Ground truth answer (yes/no/maybe).
        predicted_answer: Model's predicted answer.
        is_correct: Whether prediction matches ground truth.
        raw_response: Full LLM response text.
        sources: List of PubMed sources used.
        latency_seconds: Time taken to answer.
        error: Error message if evaluation failed.
    """
    question_id: str
    question: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    raw_response: str = ""
    sources: List[str] = field(default_factory=list)
    query_log: List[dict] = field(default_factory=list)
    reasoning_steps: List[dict] = field(default_factory=list)
    provider_used: Optional[str] = None
    latency_seconds: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question_id": self.question_id,
            "question": self.question[:200] + "..." if len(self.question) > 200 else self.question,
            "correct_answer": self.correct_answer,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "sources": self.sources,
            "provider_used": self.provider_used,
            "latency_seconds": round(self.latency_seconds, 2),
            "error": self.error,
            "raw_llm_response": self.raw_response,
            "query_log": self.query_log,
            "reasoning_steps": self.reasoning_steps,
        }


@dataclass 
class DatasetResults:
    """
    Aggregated results for the entire dataset evaluation.
    
    Attributes:
        dataset_name: Name of the evaluated dataset.
        total_questions: Total number of questions evaluated.
        correct_count: Number of correct predictions.
        accuracy: Accuracy percentage.
        results: List of individual evaluation results.
        start_time: When evaluation started.
        end_time: When evaluation ended.
        answer_distribution: Distribution of predicted answers.
    """
    dataset_name: str = "pubmedqa"
    total_questions: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    results: List[EvaluationResult] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    answer_distribution: Dict[str, int] = field(default_factory=dict)
    ground_truth_distribution: Dict[str, int] = field(default_factory=dict)
    
    def add_result(self, result: EvaluationResult) -> None:
        """Add a single result and update statistics."""
        self.results.append(result)
        self.total_questions = len(self.results)
        
        if result.is_correct:
            self.correct_count += 1
        
        self.accuracy = (self.correct_count / self.total_questions * 100) if self.total_questions > 0 else 0.0
        
        # Track answer distribution
        pred = result.predicted_answer.lower()
        self.answer_distribution[pred] = self.answer_distribution.get(pred, 0) + 1
        
        truth = result.correct_answer.lower()
        self.ground_truth_distribution[truth] = self.ground_truth_distribution.get(truth, 0) + 1
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 50,
            f"MIRAGE PubMedQA Evaluation Results",
            "=" * 50,
            f"Total Questions: {self.total_questions}",
            f"Correct: {self.correct_count}",
            f"Accuracy: {self.accuracy:.1f}%",
            "",
            "Predicted Answer Distribution:",
        ]
        
        for answer, count in sorted(self.answer_distribution.items()):
            pct = count / self.total_questions * 100 if self.total_questions > 0 else 0
            lines.append(f"  {answer}: {count} ({pct:.1f}%)")
        
        lines.append("")
        lines.append("Ground Truth Distribution:")
        for answer, count in sorted(self.ground_truth_distribution.items()):
            pct = count / self.total_questions * 100 if self.total_questions > 0 else 0
            lines.append(f"  {answer}: {count} ({pct:.1f}%)")
        
        if self.start_time and self.end_time:
            lines.append("")
            lines.append(f"Started: {self.start_time}")
            lines.append(f"Ended: {self.end_time}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "accuracy": round(self.accuracy, 2),
            "answer_distribution": self.answer_distribution,
            "ground_truth_distribution": self.ground_truth_distribution,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(output_path.parent),
            suffix=".tmp",
        ) as tmp_file:
            json.dump(payload, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            temp_path = Path(tmp_file.name)

        temp_path.replace(output_path)
        logger.info(f"Results saved to {path}")


class PubMedQAEvaluator:
    """
    Evaluator for running MedicalAgent against PubMedQA.
    
    Handles:
    - Running the agent on each question
    - Extracting yes/no/maybe answers from free-form responses
    - Calculating accuracy and statistics
    - Saving intermediate and final results
    """
    
    # Universal Multiple-Choice Regex Patterns
    ANSWER_PATTERNS = [
        r"(?:final\s+)?answer\s*(?:is\s*:|is|:)\s*[\"']?\b([A-Ea-e])\b[\"']?",
        r"(?:my\s+)?(?:conclusion|verdict)\s*(?:is|:)\s*[\"']?\b([A-Ea-e])\b[\"']?",
        r"(?:the\s+)?(?:correct\s+)?(?:option|answer)\s*(?:is|would be|should be)\s*[\"']?\b([A-Ea-e])\b[\"']?",
        r"(?:i\s+)?(?:choose|select|pick)\s*(?:option\s*)?[\"']?\b([A-Ea-e])\b[\"']?",
        r"option\s+([A-Ea-e])\s*(?:is\s+correct|is\s+the\s+answer)?",
        r"(?:^|\n)\s*\**\s*([A-Ea-e])\s*\**\s*(?:$|\n)",
        r"\[([A-Ea-e])\]",
        r"\*\*([A-Ea-e])\*\*",
    ]
    def __init__(
        self,
        agent: Any,  # MedicalAgent type, using Any to avoid circular import
        dataset: Optional[PubMedQADataset] = None,
        delay_seconds: float = 1.0,
        save_intermediate: bool = True,
        output_dir: str = "./results",
    ) -> None:
        """
        Initialize the evaluator.
        
        Args:
            agent: MedicalAgent instance to evaluate.
            dataset: PubMedQADataset instance. If None, will load automatically.
            delay_seconds: Delay between questions to respect rate limits.
            save_intermediate: Whether to save results after each question.
            output_dir: Directory to save results.
        """
        self.agent = agent
        self.dataset = dataset or PubMedQADataset()
        self.delay_seconds = float(os.getenv("EVAL_DELAY_SECONDS", delay_seconds))
        self.save_intermediate = os.getenv("EVAL_SAVE_INTERMEDIATE", str(save_intermediate)).lower() == "true"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "PubMedQA Evaluator initialized",
            extra={
                "delay_seconds": self.delay_seconds,
                "save_intermediate": self.save_intermediate,
            }
        )
    
    def extract_answer(self, response: str) -> str:
        """
        Strictly extract A/B/C/D/E multiple choice answer from LLM response.
        """
        response_upper = response.upper()

        # Universal multiple-choice regex patterns.
        patterns = [
            r"(?:FINAL\s+)?ANSWER\s*(?:IS|:)\s*[\"']?\b([A-E])\b[\"']?",
            r"(?:CORRECT\s+)?(?:OPTION|ANSWER)\s*(?:IS|WOULD BE|SHOULD BE)\s*[\"']?\b([A-E])\b[\"']?",
            r"OPTION\s+([A-E])",
            r"\*\*([A-E])\*\*",
            r"\[([A-E])\]",
        ]

        # 1) Prefer explicit answer declarations.
        for pattern in patterns:
            matches = re.findall(pattern, response_upper)
            if matches:
                # Models often put their final answer at the end.
                return matches[-1]

        # 2) Fallback: inspect tail and pick last standalone A-E letter.
        last_part = response_upper[-150:]
        matches = re.findall(r"\b([A-E])\b", last_part)
        if matches:
            return matches[-1]

        # 3) Fallback for models that output text labels instead of letters.
        text_patterns = [
            r"(?:FINAL\s+)?ANSWER\s*(?:IS|:)\s*(YES|NO|MAYBE)",
            r"\b(YES|NO|MAYBE)\b",
        ]
        text_to_letter = {"YES": "A", "NO": "B", "MAYBE": "C"}
        for pattern in text_patterns:
            text_matches = re.findall(pattern, response_upper)
            if text_matches:
                return text_to_letter[text_matches[-1]]

        logger.warning(
            "Could not extract a letter answer from response",
            extra={"response_preview": response[:200]}
        )
        return "UNKNOWN"
    
    async def evaluate_question(
        self,
        question: PubMedQAQuestion,
    ) -> EvaluationResult:
        """
        Evaluate a single question.
        
        Args:
            question: PubMedQAQuestion to evaluate.
        
        Returns:
            EvaluationResult with prediction and correctness.
        """
        start_time = time.time()
        
        try:
            # Format prompt and enforce multiple-choice instruction.
            base_prompt = question.to_prompt()
            prompt = base_prompt + "\n\nINSTRUCTIONS: You must end your response with exactly this phrase: 'The final answer is: [Letter]' (replace [Letter] with A, B, C, D, or E)."
            result = await self.agent.answer_query(prompt)
            
            latency = time.time() - start_time
            
            # Extract answer from response
            predicted = self.extract_answer(result.answer)
            correct = question.correct_answer.upper()
            is_correct = predicted == correct
            
            logger.info(
                f"Evaluated {question.question_id}",
                extra={
                    "predicted": predicted,
                    "correct": correct,
                    "is_correct": is_correct,
                    "latency": round(latency, 2),
                }
            )
            
            return EvaluationResult(
                question_id=question.question_id,
                question=question.question,
                correct_answer=correct,
                predicted_answer=predicted,
                is_correct=is_correct,
                raw_response=result.answer,
                sources=result.sources,
                query_log=getattr(result, "query_log", []),
                reasoning_steps=getattr(result, "reasoning_steps", []),
                provider_used=getattr(result, "provider_used", None),
                latency_seconds=latency,
            )
            
        except Exception as e:
            latency = time.time() - start_time
            logger.error(
                f"Failed to evaluate {question.question_id}",
                extra={"error": str(e)}
            )
            
            return EvaluationResult(
                question_id=question.question_id,
                question=question.question,
                correct_answer=question.correct_answer.upper(),
                predicted_answer="error",
                is_correct=False,
                latency_seconds=latency,
                error=str(e),
            )
    
    async def evaluate(
        self,
        limit: Optional[int] = None,
        sample_seed: Optional[int] = 42,
        resume_from: Optional[str] = None,
    ) -> DatasetResults:
        """
        Evaluate the agent on PubMedQA dataset.
        
        Args:
            limit: Maximum number of questions to evaluate. None for all.
            sample_seed: Random seed for sampling if limit is set.
            resume_from: Path to previous results JSON to resume from.
        
        Returns:
            DatasetResults with all evaluation results and statistics.
        """
        results = DatasetResults(
            dataset_name="pubmedqa",
            start_time=datetime.now().isoformat(),
        )
        
        # Resume from previous run if specified
        evaluated_ids = set()
        if resume_from and Path(resume_from).exists():
            with open(resume_from, "r") as f:
                prev_results = json.load(f)
            for r in prev_results.get("results", []):
                evaluated_ids.add(r["question_id"])
                results.add_result(EvaluationResult(
                    question_id=r["question_id"],
                    question=r["question"],
                    correct_answer=r["correct_answer"],
                    predicted_answer=r["predicted_answer"],
                    is_correct=r["is_correct"],
                    raw_response=r.get("raw_llm_response", r.get("raw_response", "")),
                    sources=r.get("sources", []),
                    query_log=r.get("query_log", []),
                    reasoning_steps=r.get("reasoning_steps", []),
                    provider_used=r.get("provider_used"),
                    latency_seconds=r.get("latency_seconds", 0),
                    error=r.get("error"),
                ))
            logger.info(f"Resumed from {len(evaluated_ids)} previous results")
        
        # Get questions to evaluate
        if limit:
            questions = self.dataset.sample(limit, seed=sample_seed)
        else:
            questions = list(self.dataset)
        
        # Filter out already evaluated
        questions = [q for q in questions if q.question_id not in evaluated_ids]
        
        total = len(questions) + len(evaluated_ids)
        logger.info(
            f"Starting evaluation",
            extra={"questions": len(questions), "already_done": len(evaluated_ids)}
        )
        
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"pubmedqa_eval_{timestamp}.json"
        
        for i, question in enumerate(questions):
            print("\n" + "=" * 70)
            print(f"🧪 QUESTION {len(results.results) + 1}/{total}: {question.question_id}")
            print(question.question)
            print("=" * 70)
            logger.info(
                f"Progress: {len(results.results)}/{total}",
                extra={"question_id": question.question_id}
            )
            
            # Evaluate
            result = await self.evaluate_question(question)
            results.add_result(result)
            
            # Save intermediate results
            if self.save_intermediate:
                results.save(str(output_path))
            
            # Delay to respect rate limits
            if i < len(questions) - 1:
                await asyncio.sleep(self.delay_seconds)
        
        results.end_time = datetime.now().isoformat()
        
        # Final save
        results.save(str(output_path))
        
        print(results.summary())
        
        return results
