# FILE: tests/test_evaluator.py
"""
Unit tests for the PubMedQA evaluator.

Tests answer extraction, evaluation logic, and result aggregation.
Uses mock agents to avoid API calls.
"""

import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from src.evaluation.evaluator import (
    PubMedQAEvaluator,
    EvaluationResult,
    DatasetResults,
)
from src.evaluation.pubmedqa_dataset import PubMedQAQuestion


# Mock agent response
@dataclass
class MockAgentResult:
    """Mock result from MedicalAgent."""
    answer: str
    sources: list


class MockMedicalAgent:
    """Mock MedicalAgent for testing without API calls."""
    
    def __init__(self, responses: dict[str, str] | None = None):
        """
        Initialize mock agent.
        
        Args:
            responses: Dict mapping question patterns to answers.
        """
        self.responses = responses or {}
        self.call_count = 0
    
    async def answer_query(self, query: str) -> MockAgentResult:
        """Return mock response based on query content."""
        self.call_count += 1
        
        # Check for matching pattern
        for pattern, response in self.responses.items():
            if pattern.lower() in query.lower():
                return MockAgentResult(answer=response, sources=["12345678"])
        
        # Default response
        return MockAgentResult(
            answer="Based on my analysis, the answer is **maybe**.",
            sources=["12345678"]
        )


@pytest.fixture
def mock_agent() -> MockMedicalAgent:
    """Create a mock agent with predefined responses."""
    return MockMedicalAgent(responses={
        "vitamin d": "Based on multiple studies, vitamin D deficiency is indeed associated with increased diabetes risk. The answer is **A**.",
        "aspirin": "The evidence does Bt support aspirin for all age groups. My conclusion is: B.",
        "coffee": "The relationship is Bt conclusive. Answer: C",
    })


@pytest.fixture
def sample_questions() -> list[PubMedQAQuestion]:
    """Create sample questions for testing."""
    return [
        PubMedQAQuestion(
            question_id="q1",
            question="Is vitamin D deficiency associated with diabetes?",
            options={"A": "A", "B": "B", "C": "C"},
            correct_answer="A",
            correct_answer_text="A",
        ),
        PubMedQAQuestion(
            question_id="q2",
            question="Does aspirin prevent events in all age groups?",
            options={"A": "A", "B": "B", "C": "C"},
            correct_answer="B",
            correct_answer_text="B",
        ),
        PubMedQAQuestion(
            question_id="q3",
            question="Is coffee and longevity relationship conclusive?",
            options={"A": "A", "B": "B", "C": "C"},
            correct_answer="C",
            correct_answer_text="C",
        ),
    ]


class TestAnswerExtraction:
    """Tests for answer extraction from LLM responses."""
    
    @pytest.fixture
    def evaluator(self, mock_agent: MockMedicalAgent) -> PubMedQAEvaluator:
        """Create evaluator with mock agent."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        return PubMedQAEvaluator(
            agent=mock_agent,
            dataset=mock_dataset,
            save_intermediate=False,
        )
    
    def test_extract_explicit_answer_is(self, evaluator: PubMedQAEvaluator):
        """Test extraction of 'the answer is X' format."""
        assert evaluator.extract_answer("The answer is A.") == "A"
        assert evaluator.extract_answer("The answer is B.") == "B"
        assert evaluator.extract_answer("The answer is C.") == "C"
    
    def test_extract_final_answer(self, evaluator: PubMedQAEvaluator):
        """Test extraction of 'final answer' format."""
        assert evaluator.extract_answer("My final answer is A") == "A"
        assert evaluator.extract_answer("Final answer: B") == "B"
    
    def test_extract_conclusion(self, evaluator: PubMedQAEvaluator):
        """Test extraction of 'conclusion' format."""
        assert evaluator.extract_answer("My conclusion is A.") == "A"
        assert evaluator.extract_answer("In conclusion: B") == "B"
    
    def test_extract_bold_answer(self, evaluator: PubMedQAEvaluator):
        """Test extraction of bold markdown answers."""
        assert evaluator.extract_answer("So the answer is **A**") == "A"
        assert evaluator.extract_answer("Therefore: **B**") == "B"
    
    def test_extract_bracketed_answer(self, evaluator: PubMedQAEvaluator):
        """Test extraction of bracketed answers."""
        assert evaluator.extract_answer("The answer is [A]") == "A"
        assert evaluator.extract_answer("[B]") == "B"
    
    def test_extract_option_letter(self, evaluator: PubMedQAEvaluator):
        """Test extraction when LLM answers with option letter."""
        assert evaluator.extract_answer("I choose option A") == "A"
        assert evaluator.extract_answer("The correct option is B") == "B"
        assert evaluator.extract_answer("Option C is the answer") == "C"
    
    def test_extract_last_occurrence(self, evaluator: PubMedQAEvaluator):
        """Test that last answer is taken when multiple are present."""
        response = """
        Initially I thought A, but after review...
        The evidence suggests B.
        Actually, the final answer is C.
        """
        assert evaluator.extract_answer(response) == "C"
    
    def test_extract_case_insensitive(self, evaluator: PubMedQAEvaluator):
        """Test case insensitive extraction."""
        assert evaluator.extract_answer("The answer is A") == "A"
        assert evaluator.extract_answer("The answer is B") == "B"
        assert evaluator.extract_answer("C") == "C"
    
    def test_extract_with_punctuation(self, evaluator: PubMedQAEvaluator):
        """Test extraction with various punctuation."""
        assert evaluator.extract_answer("Answer: A.") == "A"
        assert evaluator.extract_answer("Answer: B!") == "B"
        assert evaluator.extract_answer("Answer: 'C'") == "C"
    
    def test_extract_unknown(self, evaluator: PubMedQAEvaluator):
        """Test unknown result when B answer found."""
        result = evaluator.extract_answer("I cannot determine the answer from the data.")
        assert result == "UNKNOWN"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_correct_result(self):
        """Test creating a correct result."""
        result = EvaluationResult(
            question_id="q1",
            question="Test question?",
            correct_answer="A",
            predicted_answer="A",
            is_correct=True,
            latency_seconds=1.5,
        )
        
        assert result.is_correct is True
        assert result.error is None
    
    def test_incorrect_result(self):
        """Test creating an incorrect result."""
        result = EvaluationResult(
            question_id="q1",
            question="Test question?",
            correct_answer="A",
            predicted_answer="B",
            is_correct=False,
        )
        
        assert result.is_correct is False
    
    def test_to_dict(self):
        """Test serialization to dict."""
        result = EvaluationResult(
            question_id="q1",
            question="Test question?",
            correct_answer="A",
            predicted_answer="A",
            is_correct=True,
            sources=["123", "456"],
            latency_seconds=1.234,
        )
        
        d = result.to_dict()
        
        assert d["question_id"] == "q1"
        assert d["is_correct"] is True
        assert d["sources"] == ["123", "456"]
        assert d["latency_seconds"] == 1.23  # Rounded


class TestDatasetResults:
    """Tests for DatasetResults aggregation."""
    
    def test_add_result(self):
        """Test adding results and updating statistics."""
        results = DatasetResults()
        
        results.add_result(EvaluationResult(
            question_id="q1", question="Q1", correct_answer="A",
            predicted_answer="A", is_correct=True
        ))
        
        assert results.total_questions == 1
        assert results.correct_count == 1
        assert results.accuracy == 100.0
        
        results.add_result(EvaluationResult(
            question_id="q2", question="Q2", correct_answer="B",
            predicted_answer="A", is_correct=False
        ))
        
        assert results.total_questions == 2
        assert results.correct_count == 1
        assert results.accuracy == 50.0
    
    def test_answer_distribution(self):
        """Test tracking of answer distribution."""
        results = DatasetResults()
        
        for answer in ["A", "A", "B", "C"]:
            results.add_result(EvaluationResult(
                question_id=f"q{answer}",
                question="Q",
                correct_answer=answer,
                predicted_answer=answer,
                is_correct=True,
            ))
        
        assert results.answer_distribution["a"] == 2
        assert results.answer_distribution["b"] == 1
        assert results.answer_distribution["c"] == 1
    
    def test_summary_output(self):
        """Test summary generation."""
        results = DatasetResults(dataset_name="pubmedqa")
        results.add_result(EvaluationResult(
            question_id="q1", question="Q1", correct_answer="A",
            predicted_answer="A", is_correct=True
        ))
        
        summary = results.summary()
        
        assert "PubMedQA" in summary
        assert "Total Questions: 1" in summary
        assert "Correct: 1" in summary
        assert "100.0%" in summary


class TestPubMedQAEvaluator:
    """Integration tests for the evaluator."""
    
    @pytest.fixture
    def mock_dataset(self, sample_questions: list) -> MagicMock:
        """Create mock dataset."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=len(sample_questions))
        dataset.__iter__ = MagicMock(return_value=iter(sample_questions))
        dataset.sample = MagicMock(return_value=sample_questions)
        return dataset
    
    @pytest.mark.asyncio
    async def test_evaluate_single_question(
        self,
        mock_agent: MockMedicalAgent,
        sample_questions: list,
    ):
        """Test evaluating a single question."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        
        evaluator = PubMedQAEvaluator(
            agent=mock_agent,
            dataset=mock_dataset,
            save_intermediate=False,
        )
        
        question = sample_questions[0]  # vitamin D question, answer=yes
        result = await evaluator.evaluate_question(question)
        
        assert result.question_id == "q1"
        assert result.predicted_answer == "A"
        assert result.is_correct is True
        assert result.latency_seconds >= 0  # Mock is instant, so >= 0
    
    @pytest.mark.asyncio
    async def test_evaluate_handles_error(self):
        """Test that errors are caught and recorded."""
        # Create agent that raises error
        failing_agent = MagicMock()
        failing_agent.answer_query = AsyncMock(side_effect=Exception("API Error"))
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        
        evaluator = PubMedQAEvaluator(
            agent=failing_agent,
            dataset=mock_dataset,
            save_intermediate=False,
        )
        
        question = PubMedQAQuestion(
            question_id="error_q",
            question="Will this fail?",
            options={"A": "A", "B": "B", "C": "C"},
            correct_answer="A",
            correct_answer_text="A",
        )
        
        result = await evaluator.evaluate_question(question)
        
        assert result.is_correct is False
        assert result.predicted_answer == "error"
        assert "API Error" in result.error
