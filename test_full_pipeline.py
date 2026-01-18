# FILE: test_full_pipeline.py
"""Test the full agentic RAG pipeline with real LLMs."""

import logging
import os
import sys
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("="*70)
print("🧪 FULL AGENTIC RAG PIPELINE TEST")
print("="*70)

# Check API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"✅ GOOGLE_API_KEY found ({api_key[:10]}...)")
else:
    print("⚠️  GOOGLE_API_KEY not set - will try Ollama")

# Import components
from src.orchestrator.state import create_initial_state, GraphState
from src.orchestrator.graph import build_agentic_graph, run_query_sync
from src.agents.planner import planner_node_sync
from src.agents.retriever import retriever_node_sync
from src.agents.extractor import extractor_node_sync
from src.agents.synthesizer import synthesizer_node_sync

# Build graph with real agents
print("\n🔧 Building graph with real agent implementations...")
graph = build_agentic_graph(
    planner=planner_node_sync,
    retriever=retriever_node_sync,
    extractor=extractor_node_sync,
    synthesizer=synthesizer_node_sync,
)
print("✅ Graph built successfully")

# Test question
question = "What are the latest immunotherapy treatments for advanced melanoma?"
print(f"\n📋 Question: {question}")
print("-"*70)

# Run query
print("\n⏳ Running query (this may take 30-90 seconds)...")
try:
    result = run_query_sync(graph, question, max_iterations=2)
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    # Plan
    plan = result.get("plan", [])
    print(f"\n📝 Plan ({len(plan)} steps):")
    for i, step in enumerate(plan[:5], 1):
        print(f"   {i}. {step[:80]}")
    
    # PICO Query
    pico = result.get("pico_query", {})
    if pico:
        print(f"\n🎯 PICO Query:")
        print(f"   Population: {pico.get('population', [])}")
        print(f"   Intervention: {pico.get('intervention', [])}")
        print(f"   Modifiers: {pico.get('modifiers', [])}")
    
    # Retrieved docs
    docs = result.get("retrieved_docs", [])
    print(f"\n📚 Retrieved Documents ({len(docs)}):")
    for doc in docs[:3]:
        print(f"   - PMID:{doc['pmid']}: {doc['title'][:50]}...")
    
    # Extracted notes
    notes = result.get("extracted_notes", [])
    print(f"\n📋 Extracted Notes ({len(notes)}):")
    for note in notes[:2]:
        print(f"   {note[:100]}...")
    
    # Final answer
    answer = result.get("final_answer", "No answer generated")
    print(f"\n💡 Final Answer:")
    print("-"*70)
    print(answer[:800])
    if len(answer) > 800:
        print("...")
    
    # Metrics
    print(f"\n📈 Metrics:")
    print(f"   Confidence: {result.get('final_confidence', 0)}/10")
    print(f"   Iterations: {result.get('iteration_count', 0)}")
    print(f"   Reasoning Steps: {len(result.get('reasoning_trace', []))}")
    
    # Reasoning trace
    print(f"\n🧠 Reasoning Trace:")
    for step in result.get("reasoning_trace", [])[:6]:
        print(f"   [{step.get('phase')}] {step.get('thought', '')[:60]}")
    
    print("\n" + "="*70)
    print("✅ FULL PIPELINE TEST COMPLETE")
    print("="*70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
