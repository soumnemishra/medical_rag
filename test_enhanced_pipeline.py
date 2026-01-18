# FILE: test_enhanced_pipeline.py
"""Test the enhanced agentic RAG pipeline with MA-RAG style step execution."""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("="*70)
print("🧪 ENHANCED AGENTIC RAG PIPELINE TEST")
print("="*70)

# Build enhanced graph
print("\n🔧 Building enhanced graph with step executor...")
from src.orchestrator.graph import build_enhanced_graph, run_query_sync
from src.orchestrator.state import create_initial_state

graph = build_enhanced_graph()
print("✅ Enhanced graph built successfully")

# Test question
question = "What are the immunotherapy options for melanoma?"
print(f"\n📋 Question: {question}")
print("-"*70)

# Run query
print("\n⏳ Running enhanced pipeline (step-by-step execution)...")
try:
    initial_state = create_initial_state(question)
    initial_state["max_iterations"] = 2
    
    result = graph.invoke(initial_state)
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    # Plan
    plan = result.get("plan", [])
    print(f"\n📝 Plan ({len(plan)} steps):")
    for i, step in enumerate(plan, 1):
        print(f"   {i}. {step[:80]}")
    
    # Step outputs
    step_outputs = result.get("step_outputs", [])
    print(f"\n📋 Step Outputs ({len(step_outputs)}):")
    for step in step_outputs[:3]:
        if isinstance(step, dict):
            step_name = step.get("step_name", "Step")
            task = step.get("task", "")[:50]
            answer = step.get("answer", {})
            conf = answer.get("confidence", 0) if isinstance(answer, dict) else 5
            print(f"   [{step_name}] {task}... (conf: {conf}/10)")
    
    # Retrieved docs
    docs = result.get("retrieved_docs", [])
    print(f"\n📚 Total Retrieved Documents: {len(docs)}")
    for doc in docs[:3]:
        print(f"   - PMID:{doc['pmid']}: {doc['title'][:50]}...")
    
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
    print(f"   Plan Steps: {len(plan)}")
    print(f"   Step Outputs: {len(step_outputs)}")
    print(f"   Documents: {len(docs)}")
    
    # Reasoning trace
    print(f"\n🧠 Reasoning Trace:")
    for step in result.get("reasoning_trace", [])[:6]:
        phase = step.get("phase", "?")
        thought = step.get("thought", "")[:60]
        print(f"   [{phase}] {thought}")
    
    print("\n" + "="*70)
    print("✅ ENHANCED PIPELINE TEST COMPLETE")
    print("="*70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
