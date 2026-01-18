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
        "vitamin d": "Based on multiple studies, vitamin D deficiency is indeed associated with increased diabetes risk. The answer is **yes**.",
        "aspirin": "The evidence does not support aspirin for all age groups. My conclusion is: no.",
        "coffee": "The relationship is not conclusive. Answer: maybe",
    })


@pytest.fixture
def sample_questions() -> list[PubMedQAQuestion]:
    """Create sample questions for testing."""
    return [
        PubMedQAQuestion(
            question_id="q1",
            question="Is vitamin D deficiency associated with diabetes?",
            options={"A": "yes", "B": "no", "C": "maybe"},
            correct_answer="A",
            correct_answer_text="yes",
        ),
        PubMedQAQuestion(
            question_id="q2",
            question="Does aspirin prevent events in all age groups?",
            options={"A": "yes", "B": "no", "C": "maybe"},
            correct_answer="B",
            correct_answer_text="no",
        ),
        PubMedQAQuestion(
            question_id="q3",
            question="Is coffee and longevity relationship conclusive?",
            options={"A": "yes", "B": "no", "C": "maybe"},
            correct_answer="C",
            correct_answer_text="maybe",
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
        assert evaluator.extract_answer("The answer is yes.") == "yes"
        assert evaluator.extract_answer("The answer is no.") == "no"
        assert evaluator.extract_answer("The answer is maybe.") == "maybe"
    
    def test_extract_final_answer(self, evaluator: PubMedQAEvaluator):
        """Test extraction of 'final answer' format."""
        assert evaluator.extract_answer("My final answer is yes") == "yes"
        assert evaluator.extract_answer("Final answer: no") == "no"
    
    def test_extract_conclusion(self, evaluator: PubMedQAEvaluator):
        """Test extraction of 'conclusion' format."""
        assert evaluator.extract_answer("My conclusion is yes.") == "yes"
        assert evaluator.extract_answer("In conclusion: no") == "no"
    
    def test_extract_bold_answer(self, evaluator: PubMedQAEvaluator):
        """Test extraction of bold markdown answers."""
        assert evaluator.extract_answer("So the answer is **yes**") == "yes"
        assert evaluator.extract_answer("Therefore: **no**") == "no"
    
    def test_extract_bracketed_answer(self, evaluator: PubMedQAEvaluator):
        """Test extraction of bracketed answers."""
        assert evaluator.extract_answer("The answer is [yes]") == "yes"
        assert evaluator.extract_answer("[no]") == "no"
    
    def test_extract_option_letter(self, evaluator: PubMedQAEvaluator):
        """Test extraction when LLM answers with option letter."""
        assert evaluator.extract_answer("I choose option A") == "yes"
        assert evaluator.extract_answer("The correct option is B") == "no"
        assert evaluator.extract_answer("Option C is the answer") == "maybe"
    
    def test_extract_last_occurrence(self, evaluator: PubMedQAEvaluator):
        """Test that last answer is taken when multiple are present."""
        response = """
        Initially I thought yes, but after review...
        The evidence suggests no.
        Actually, the final answer is maybe.
        """
        assert evaluator.extract_answer(response) == "maybe"
    
    def test_extract_case_insensitive(self, evaluator: PubMedQAEvaluator):
        """Test case insensitive extraction."""
        assert evaluator.extract_answer("The answer is YES") == "yes"
        assert evaluator.extract_answer("The answer is NO") == "no"
        assert evaluator.extract_answer("MAYBE") == "maybe"
    
    def test_extract_with_punctuation(self, evaluator: PubMedQAEvaluator):
        """Test extraction with various punctuation."""
        assert evaluator.extract_answer("Answer: yes.") == "yes"
        assert evaluator.extract_answer("Answer: no!") == "no"
        assert evaluator.extract_answer("Answer: 'maybe'") == "maybe"
    
    def test_extract_fallback_counting(self, evaluator: PubMedQAEvaluator):
        """Test fallback counting when no pattern matches."""
        response = "Yes, this is true. Yes, confirmed. Yes."
        # Should count and return most frequent in last part
        result = evaluator.extract_answer(response)
        assert result == "yes"
    
    def test_extract_unknown(self, evaluator: PubMedQAEvaluator):
        """Test unknown result when no answer found."""
        result = evaluator.extract_answer("I cannot determine the answer from the data.")
        assert result == "unknown"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_correct_result(self):
        """Test creating a correct result."""
        result = EvaluationResult(
            question_id="q1",
            question="Test question?",
            correct_answer="yes",
            predicted_answer="yes",
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
            correct_answer="yes",
            predicted_answer="no",
            is_correct=False,
        )
        
        assert result.is_correct is False
    
    def test_to_dict(self):
        """Test serialization to dict."""
        result = EvaluationResult(
            question_id="q1",
            question="Test question?",
            correct_answer="yes",
            predicted_answer="yes",
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
            question_id="q1", question="Q1", correct_answer="yes",
            predicted_answer="yes", is_correct=True
        ))
        
        assert results.total_questions == 1
        assert results.correct_count == 1
        assert results.accuracy == 100.0
        
        results.add_result(EvaluationResult(
            question_id="q2", question="Q2", correct_answer="no",
            predicted_answer="yes", is_correct=False
        ))
        
        assert results.total_questions == 2
        assert results.correct_count == 1
        assert results.accuracy == 50.0
    
    def test_answer_distribution(self):
        """Test tracking of answer distribution."""
        results = DatasetResults()
        
        for answer in ["yes", "yes", "no", "maybe"]:
            results.add_result(EvaluationResult(
                question_id=f"q{answer}",
                question="Q",
                correct_answer=answer,
                predicted_answer=answer,
                is_correct=True,
            ))
        
        assert results.answer_distribution["yes"] == 2
        assert results.answer_distribution["no"] == 1
        assert results.answer_distribution["maybe"] == 1
    
    def test_summary_output(self):
        """Test summary generation."""
        results = DatasetResults(dataset_name="pubmedqa")
        results.add_result(EvaluationResult(
            question_id="q1", question="Q1", correct_answer="yes",
            predicted_answer="yes", is_correct=True
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
        assert result.predicted_answer == "yes"
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
            options={"A": "yes", "B": "no", "C": "maybe"},
            correct_answer="A",
            correct_answer_text="yes",
        )
        
        result = await evaluator.evaluate_question(question)
        
        assert result.is_correct is False
        assert result.predicted_answer == "error"
        assert "API Error" in result.error
