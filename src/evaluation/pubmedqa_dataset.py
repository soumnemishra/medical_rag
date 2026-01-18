# FILE: src/evaluation/pubmedqa_dataset.py
"""
PubMedQA Dataset Loader for MIRAGE Benchmark.

Downloads and parses the PubMedQA portion of the MIRAGE benchmark.
PubMedQA* contains 500 expert-annotated questions with yes/no/maybe answers.

Example Usage:
    >>> from src.evaluation.pubmedqa_dataset import PubMedQADataset
    >>> dataset = PubMedQADataset()
    >>> print(len(dataset))
    500
    >>> print(dataset[0])
    PubMedQAQuestion(question_id='pubmedqa_0', question='...', answer='yes')

ENV_VARS:
    MIRAGE_DATA_PATH: Path to benchmark.json (default: ./data/benchmark.json)
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.exceptions import IngestError

logger = logging.getLogger(__name__)

# Google Drive direct download URL for benchmark.json
BENCHMARK_GDRIVE_ID = "1ryvimxhOJXVGpYEIY_eak9X_YVWz1Axd"
BENCHMARK_URL = f"https://drive.google.com/uc?export=download&id={BENCHMARK_GDRIVE_ID}"

# GitHub raw URL as fallback
BENCHMARK_GITHUB_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"


@dataclass
class PubMedQAQuestion:
    """
    A single PubMedQA question from the MIRAGE benchmark.
    
    Attributes:
        question_id: Unique identifier for the question.
        question: The question text (without context, per MIRAGE rules).
        options: Answer options {"A": "yes", "B": "no", "C": "maybe"}.
        correct_answer: The correct option key ("A", "B", or "C").
        correct_answer_text: The correct answer text ("yes", "no", or "maybe").
    """
    question_id: str
    question: str
    options: Dict[str, str]
    correct_answer: str  # "A", "B", or "C"
    correct_answer_text: str  # "yes", "no", or "maybe"
    
    def to_prompt(self) -> str:
        """
        Format question as a prompt for the MedicalAgent.
        
        Returns:
            Formatted prompt string with question and options.
        """
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(self.options.items())])
        return f"""Based on scientific literature, answer the following biomedical research question.

Question: {self.question}

Options:
{options_text}

INSTRUCTIONS:
1. Search PubMed for relevant literature on this topic.
2. Analyze the evidence from the retrieved articles.
3. Make a decisive judgment based on the preponderance of evidence:
   - Answer "yes" if the evidence SUPPORTS the claim
   - Answer "no" if the evidence CONTRADICTS or does not support the claim
   - Answer "maybe" ONLY if the evidence is truly mixed, insufficient, or inconclusive
4. Do NOT default to "maybe" out of caution - commit to yes or no when the evidence leans one way.

After your analysis, state your final answer clearly as: **Final Answer: [yes/no/maybe]**"""


class PubMedQADataset:
    """
    Loader for PubMedQA portion of the MIRAGE benchmark.
    
    The dataset contains 500 expert-annotated biomedical questions
    requiring yes/no/maybe answers based on scientific literature.
    
    Attributes:
        questions: List of PubMedQAQuestion objects.
        data_path: Path to the benchmark.json file.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        auto_download: bool = True,
    ) -> None:
        """
        Initialize the PubMedQA dataset.
        
        Args:
            data_path: Path to benchmark.json. Defaults to ./data/benchmark.json
            auto_download: If True, download benchmark.json if not present.
        
        Raises:
            IngestError: If data file not found and auto_download is False.
        """
        self.data_path = Path(data_path or os.getenv(
            "MIRAGE_DATA_PATH", 
            "./data/benchmark.json"
        ))
        self.questions: List[PubMedQAQuestion] = []
        
        # Ensure data directory exists
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download if needed
        if not self.data_path.exists():
            if auto_download:
                self._download_benchmark()
            else:
                raise IngestError(
                    f"Benchmark file not found at {self.data_path}. "
                    "Set auto_download=True or download manually."
                )
        
        # Load the dataset
        self._load_pubmedqa()
        
        logger.info(
            "PubMedQA dataset loaded",
            extra={"num_questions": len(self.questions)}
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _download_benchmark(self) -> None:
        """
        Download benchmark.json from GitHub or Google Drive.
        
        Uses GitHub raw URL as primary (more reliable), with Drive as fallback.
        
        Raises:
            IngestError: If download fails after retries.
        """
        logger.info("Downloading MIRAGE benchmark.json...")
        
        try:
            # Try GitHub first (more reliable)
            response = requests.get(BENCHMARK_GITHUB_URL, timeout=60)
            response.raise_for_status()
            
            # Validate JSON
            data = response.json()
            if "pubmedqa" not in data:
                raise ValueError("Invalid benchmark format: missing 'pubmedqa' key")
            
            # Save to disk
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            logger.info(
                "Benchmark downloaded successfully",
                extra={"path": str(self.data_path), "size_mb": len(response.content) / 1024 / 1024}
            )
            
        except Exception as e:
            logger.error("Failed to download benchmark", extra={"error": str(e)})
            raise IngestError(f"Failed to download benchmark: {e}") from e
    
    def _load_pubmedqa(self) -> None:
        """
        Load and parse PubMedQA questions from benchmark.json.
        
        The MIRAGE benchmark stores PubMedQA as a dict keyed by PMID.
        Each item has: question, options, answer (A/B/C), and PMID list.
        """
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                benchmark = json.load(f)
            
            pubmedqa_data = benchmark.get("pubmedqa", {})
            
            if not pubmedqa_data:
                raise IngestError("No PubMedQA data found in benchmark.json")
            
            # Standard options for PubMedQA (yes/no/maybe)
            standard_options = {
                "A": "yes",
                "B": "no", 
                "C": "maybe"
            }
            
            # Mapping from option letter to answer text
            option_to_text = {"A": "yes", "B": "no", "C": "maybe"}
            
            # Handle both dict (keyed by PMID) and list formats
            if isinstance(pubmedqa_data, dict):
                items = list(pubmedqa_data.items())
            else:
                items = [(f"item_{i}", item) for i, item in enumerate(pubmedqa_data)]
            
            for idx, (pmid, item) in enumerate(items):
                # Get the question and answer
                question_text = item.get("question", "")
                answer_key = item.get("answer", "C").upper()  # A, B, or C
                
                # Get options, defaulting to standard if not provided
                options = item.get("options", standard_options)
                if not options:
                    options = standard_options
                
                # Get the answer text from the option key
                correct_answer_text = option_to_text.get(answer_key, "maybe")
                
                self.questions.append(PubMedQAQuestion(
                    question_id=f"pubmedqa_{pmid}",
                    question=question_text,
                    options=options if isinstance(options, dict) else standard_options,
                    correct_answer=answer_key,
                    correct_answer_text=correct_answer_text,
                ))
            
            logger.info(
                "Parsed PubMedQA questions",
                extra={"count": len(self.questions)}
            )
            
        except json.JSONDecodeError as e:
            raise IngestError(f"Invalid JSON in benchmark file: {e}") from e
        except KeyError as e:
            raise IngestError(f"Missing required field in benchmark: {e}") from e
    
    def __len__(self) -> int:
        """Return number of questions."""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> PubMedQAQuestion:
        """Get question by index."""
        return self.questions[idx]
    
    def __iter__(self) -> Iterator[PubMedQAQuestion]:
        """Iterate over questions."""
        return iter(self.questions)
    
    def sample(self, n: int, seed: Optional[int] = None) -> List[PubMedQAQuestion]:
        """
        Get a random sample of questions.
        
        Args:
            n: Number of questions to sample.
            seed: Random seed for reproducibility.
        
        Returns:
            List of sampled PubMedQAQuestion objects.
        """
        import random
        
        if seed is not None:
            random.seed(seed)
        
        n = min(n, len(self.questions))
        return random.sample(self.questions, n)
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get dataset statistics.
        
        Returns:
            Dict with counts of questions by answer type.
        """
        stats = {"yes": 0, "no": 0, "maybe": 0, "total": len(self.questions)}
        
        for q in self.questions:
            answer = q.correct_answer_text.lower()
            if answer in stats:
                stats[answer] += 1
        
        return stats
