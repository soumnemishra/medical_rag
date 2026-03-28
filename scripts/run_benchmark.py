# FILE: scripts/run_benchmark.py
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import MedicalAgent
from src.evaluation.evaluator import PubMedQAEvaluator


def configure_run_logging(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "pubmedqa_run.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path)
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return log_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MIRAGE PubMedQA benchmark.")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of questions to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Delay between questions in seconds. Overrides EVAL_DELAY_SECONDS when provided.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a previous results JSON file to resume from.",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        type=str,
        default="./results/benchmark",
        help="Directory where benchmark results will be written.",
    )
    return parser.parse_args()


async def run_benchmark() -> None:
    args = parse_args()

    if args.delay is not None:
        os.environ["EVAL_DELAY_SECONDS"] = str(args.delay)

    log_path = configure_run_logging(args.output_dir)
    logging.getLogger(__name__).info(
        "Benchmark logging initialized",
        extra={"log_path": str(log_path)}
    )

    print("Starting MIRAGE PubMedQA benchmark...")
    print(f"📄 Run log: {log_path}")
    agent = MedicalAgent()
    evaluator = PubMedQAEvaluator(
        agent=agent,
        output_dir=args.output_dir,
    )

    try:
        results = await evaluator.evaluate(
            limit=args.sample_size,
            sample_seed=args.seed,
            resume_from=args.resume_from,
        )
    finally:
        agent.close()

    grounded_count = sum(1 for r in results.results if len(r.sources) > 0)
    grounded_rate = (grounded_count / results.total_questions * 100) if results.total_questions else 0.0

    print("\n--- BENCHMARK METRICS ---")
    print(f"Grounded Coverage: {grounded_rate:.1f}%")
    print(f"Parsed Results: {results.total_questions - sum(1 for r in results.results if r.predicted_answer == 'UNKNOWN')}")
    print(f"Unknown Parses: {sum(1 for r in results.results if r.predicted_answer == 'UNKNOWN')}")

    if results.total_questions and grounded_rate < 80.0:
        print("WARNING: Model is failing to use the PubMed tool reliably.")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
