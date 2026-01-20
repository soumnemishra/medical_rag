import asyncio
import logging
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.adapter import MaRagAdapter
from src.evaluation.evaluator import PubMedQAEvaluator
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')

async def main():
    print("Initializing Evaluation...")
    
    # 1. Initialize Adapter
    agent = MaRagAdapter()
    
    # 2. Initialize Evaluator
    # Evaluator will auto-load PubMedQA dataset
    evaluator = PubMedQAEvaluator(
        agent=agent,
        output_dir="results",
        delay_seconds=2.0, # Give Ollama breathing room
        save_intermediate=True
    )
    
    # 3. Run Evaluation
    print("Starting Pilot Evaluation (Limit: 5 samples)...")
    results = await evaluator.evaluate(limit=5)
    
    print("\nEvaluation Complete.")
    print(results.summary())

if __name__ == "__main__":
    asyncio.run(main())
