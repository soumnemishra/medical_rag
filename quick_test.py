# FILE: quick_test.py
"""Quick test without input prompts."""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("="*60)
print("TEST 1: Placeholder Graph")
print("="*60)

from src.orchestrator.graph import build_agentic_graph, run_query_sync

graph = build_agentic_graph()
result = run_query_sync(graph, "What is the treatment for melanoma?")

print(f"Plan: {result.get('plan', [])}")
print(f"Confidence: {result.get('final_confidence', 0)}/10")
print(f"Iterations: {result.get('iteration_count', 0)}")
print(f"Answer: {result.get('final_answer', 'N/A')[:80]}...")

print("\n" + "="*60)
print("TEST 2: Retriever Agent (PubMed)")
print("="*60)

from src.agents.retriever import RetrieverAgent
from src.orchestrator.state import create_initial_state
import asyncio

retriever = RetrieverAgent(max_results=3)
state = create_initial_state("melanoma treatment")
state["pico_query"] = {
    "population": ["melanoma"],
    "intervention": ["treatment"],
    "modifiers": [],
}

result2 = asyncio.run(retriever.retrieve(state))
docs = result2.get("retrieved_docs", [])
print(f"Retrieved {len(docs)} documents:")
for doc in docs:
    print(f"  - PMID:{doc['pmid']}: {doc['title'][:50]}...")

print("\n✅ Tests complete!")
