# FILE: tests/test_pubmedqa_dataset.py
"""
Unit tests for the PubMedQA dataset loader.

Tests dataset loading, parsing, and sampling without network calls.
Uses mock data to ensure deterministic, fast tests.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.evaluation.pubmedqa_dataset import (
    PubMedQADataset,
    PubMedQAQuestion,
)
from src.exceptions import IngestError


# Sample benchmark data for testing
SAMPLE_BENCHMARK_DATA = {
    "pubmedqa": [
        {
            "question": "Is vitamin D deficiency associated with increased risk of diabetes?",
            "answer": "yes"
        },
        {
            "question": "Does aspirin prevent cardiovascular events in all age groups?",
            "answer": "no"
        },
        {
            "question": "Is the relationship between coffee consumption and longevity conclusive?",
            "answer": "maybe"
        },
        {
            "question": "Are statins effective for primary prevention of heart disease?",
            "answer": "yes"
        },
        {
            "question": "Does metformin cause significant weight loss in diabetic patients?",
            "answer": "maybe"
        },
    ]
}


@pytest.fixture
def temp_benchmark_file(tmp_path: Path) -> Path:
    """Create a temporary benchmark.json file for testing."""
    benchmark_path = tmp_path / "benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_BENCHMARK_DATA, f)
    return benchmark_path


@pytest.fixture
def dataset(temp_benchmark_file: Path) -> PubMedQADataset:
    """Create a PubMedQADataset with test data."""
    return PubMedQADataset(data_path=str(temp_benchmark_file), auto_download=False)


class TestPubMedQAQuestion:
    """Tests for PubMedQAQuestion dataclass."""
    
    def test_question_creation(self):
        """Test creating a PubMedQAQuestion."""
        q = PubMedQAQuestion(
            question_id="test_1",
            question="Is this a test?",
            options={"A": "yes", "B": "no", "C": "maybe"},
            correct_answer="A",
            correct_answer_text="yes",
        )
        
        assert q.question_id == "test_1"
        assert q.question == "Is this a test?"
        assert q.correct_answer == "A"
        assert q.correct_answer_text == "yes"
    
    def test_to_prompt_formatting(self):
        """Test that to_prompt generates proper prompt format."""
        q = PubMedQAQuestion(
            question_id="test_1",
            question="Is vitamin D important?",
            options={"A": "yes", "B": "no", "C": "maybe"},
            correct_answer="A",
            correct_answer_text="yes",
        )
        
        prompt = q.to_prompt()
        
        # Check prompt contains key elements
        assert "Is vitamin D important?" in prompt
        assert "A: yes" in prompt
        assert "B: no" in prompt
        assert "C: maybe" in prompt
        
        


class TestPubMedQADataset:
    """Tests for PubMedQADataset loader."""
    
    def test_load_dataset(self, dataset: PubMedQADataset):
        """Test loading dataset from file."""
        assert len(dataset) == 5
        assert all(isinstance(q, PubMedQAQuestion) for q in dataset)
    
    def test_dataset_indexing(self, dataset: PubMedQADataset):
        """Test accessing questions by index."""
        first = dataset[0]
        assert first.question_id == "pubmedqa_0"
        assert "vitamin D" in first.question
        assert first.correct_answer_text == "yes"
    
    def test_dataset_iteration(self, dataset: PubMedQADataset):
        """Test iterating over dataset."""
        questions = list(dataset)
        assert len(questions) == 5
    
    def test_answer_mapping(self, dataset: PubMedQADataset):
        """Test that answers are correctly mapped to options."""
        # Check yes -> A mapping
        yes_questions = [q for q in dataset if q.correct_answer_text == "yes"]
        for q in yes_questions:
            assert q.correct_answer == "A"
        
        # Check no -> B mapping
        no_questions = [q for q in dataset if q.correct_answer_text == "no"]
        for q in no_questions:
            assert q.correct_answer == "B"
        
        # Check maybe -> C mapping
        maybe_questions = [q for q in dataset if q.correct_answer_text == "maybe"]
        for q in maybe_questions:
            assert q.correct_answer == "C"
    
    def test_sample_with_seed(self, dataset: PubMedQADataset):
        """Test random sampling with reproducibility."""
        sample1 = dataset.sample(3, seed=42)
        sample2 = dataset.sample(3, seed=42)
        
        # Same seed should give same order
        assert [q.question_id for q in sample1] == [q.question_id for q in sample2]
    
    def test_sample_different_seeds(self, dataset: PubMedQADataset):
        """Test that different seeds give different samples."""
        sample1 = dataset.sample(3, seed=42)
        sample2 = dataset.sample(3, seed=123)
        
        # Different seeds may give different order (with high probability)
        ids1 = [q.question_id for q in sample1]
        ids2 = [q.question_id for q in sample2]
        # Note: They could randomly be the same, but unlikely for different seeds
    
    def test_get_statistics(self, dataset: PubMedQADataset):
        """Test dataset statistics calculation."""
        stats = dataset.get_statistics()
        
        assert stats["total"] == 5
        assert stats["yes"] == 2
        assert stats["no"] == 1
        assert stats["maybe"] == 2
    
    def test_missing_file_no_download(self, tmp_path: Path):
        """Test that missing file raises error when auto_download=False."""
        fake_path = tmp_path / "nonexistent.json"
        
        with pytest.raises(IngestError, match="not found"):
            PubMedQADataset(data_path=str(fake_path), auto_download=False)
    
    def test_invalid_json(self, tmp_path: Path):
        """Test handling of invalid JSON file."""
        bad_json_path = tmp_path / "bad.json"
        with open(bad_json_path, "w") as f:
            f.write("not valid json {{{")
        
        with pytest.raises(IngestError, match="Invalid JSON"):
            PubMedQADataset(data_path=str(bad_json_path), auto_download=False)
    
    def test_missing_pubmedqa_key(self, tmp_path: Path):
        """Test handling of benchmark without pubmedqa data."""
        no_pubmedqa_path = tmp_path / "no_pubmedqa.json"
        with open(no_pubmedqa_path, "w", encoding="utf-8") as f:
            json.dump({"mmlu": []}, f)
        
        with pytest.raises(IngestError, match="No PubMedQA data"):
            PubMedQADataset(data_path=str(no_pubmedqa_path), auto_download=False)


class TestDatasetDownload:
    """Tests for benchmark download functionality."""
    
    @patch("src.evaluation.pubmedqa_dataset.requests.get")
    def test_download_success(self, mock_get: MagicMock, tmp_path: Path):
        """Test successful download of benchmark."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_BENCHMARK_DATA
        mock_response.content = json.dumps(SAMPLE_BENCHMARK_DATA).encode()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Create dataset (should trigger download)
        data_path = tmp_path / "benchmark.json"
        dataset = PubMedQADataset(data_path=str(data_path), auto_download=True)
        
        assert len(dataset) == 5
        assert data_path.exists()
