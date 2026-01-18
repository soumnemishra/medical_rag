# FILE: scripts/run_benchmark.py
"""
CLI Script to run PubMedQA benchmark evaluation.

Usage:
    # Run on 50 random questions (recommended for testing)
    python scripts/run_benchmark.py --sample-size 50
    
    # Run on all 500 questions
    python scripts/run_benchmark.py --all
    
    # Run with verbose output and custom output directory
    python scripts/run_benchmark.py --sample-size 100 --verbose --output ./my_results
    
    # Resume a previous interrupted run
    python scripts/run_benchmark.py --resume ./results/pubmedqa_eval_20260116.json

ENV_VARS:
    GOOGLE_CLOUD_PROJECT: GCP project ID for Vertex AI
    GOOGLE_CLOUD_LOCATION: GCP location (default: us-central1)
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.agent import MedicalAgent # imported the agents 
from src.evaluation.pubmedqa_dataset import PubMedQADataset # imported the dataset 
from src.evaluation.evaluator import PubMedQAEvaluator # imported the evaluator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the benchmark run."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Reduce noise from HTTP libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PubMedQA benchmark evaluation for MedicalAgent RAG chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py --sample-size 50           # Quick test with 50 questions
  python scripts/run_benchmark.py --sample-size 100 --seed 123  # Reproducible sample
  python scripts/run_benchmark.py --all                      # Full benchmark (500 questions)
  python scripts/run_benchmark.py --resume results/prev.json # Resume interrupted run
        """,
    )
    
    # Evaluation scope
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sample-size", "-n",
        type=int,
        default=50,
        help="Number of questions to sample (default: 50)",
    )
    group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run on all 500 questions",
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    
    # Resume
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to previous results JSON to resume from",
    )
    
    # Rate limiting
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between questions in seconds (default: 1.0)",
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    # Data path
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to benchmark.json (default: ./data/benchmark.json)",
    )
    
    return parser.parse_args()


async def main() -> int:
    """Main entry point for benchmark evaluation."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("MIRAGE PubMedQA Benchmark Evaluation")
    print("=" * 60)
    
    # Check for required environment variables
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        print("\n❌ ERROR: GOOGLE_CLOUD_PROJECT environment variable not set.")
        print("   Set it in your .env file or export it in your shell.")
        return 1
    
    try:
        # Initialize components
        print("\n📦 Loading PubMedQA dataset...")
        dataset = PubMedQADataset(data_path=args.data_path)
        stats = dataset.get_statistics()
        print(f"   Loaded {stats['total']} questions")
        print(f"   Distribution: yes={stats['yes']}, no={stats['no']}, maybe={stats['maybe']}")
        
        print("\n🤖 Initializing MedicalAgent...")
        agent = MedicalAgent()
        print(f"   Using Vertex AI with Gemini")
        
        print("\n📊 Setting up evaluator...")
        evaluator = PubMedQAEvaluator(
            agent=agent,
            dataset=dataset,
            delay_seconds=args.delay,
            output_dir=args.output,
        )
        
        # Determine sample size
        sample_size = None if args.all else args.sample_size
        
        if sample_size:
            print(f"\n🎯 Will evaluate {sample_size} random questions (seed={args.seed})")
        else:
            print(f"\n🎯 Will evaluate ALL {len(dataset)} questions")
        
        if args.resume:
            print(f"📂 Resuming from: {args.resume}")
        
        # Confirm before starting full benchmark
        if not sample_size or sample_size > 100:
            print("\n⚠️  This may take a while and consume API credits.")
            response = input("   Continue? [y/N]: ").strip().lower()
            if response != "y":
                print("   Aborted.")
                return 0
        
        print("\n🚀 Starting evaluation...\n")
        
        # Run evaluation
        results = await evaluator.evaluate(
            limit=sample_size,
            sample_seed=args.seed,
            resume_from=args.resume,
        )
        
        print("\n✅ Evaluation complete!")
        print(f"   Results saved to: {args.output}/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Partial results may have been saved.")
        return 130
        
    except Exception as e:
        logger.exception("Benchmark failed")
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
