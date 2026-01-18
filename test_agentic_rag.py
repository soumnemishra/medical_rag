# FILE: test_agentic_rag.py
"""
Quick test script for the Agentic RAG implementation.
Run this to see the full pipeline in action.
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def test_with_placeholder_graph():
    """Test using placeholder nodes (no LLM calls)."""
    print("\n" + "="*60)
    print("TEST 1: Placeholder Graph (no LLM calls)")
    print("="*60)
    
    from src.orchestrator.graph import build_agentic_graph, run_query_sync
    
    # Build graph with placeholder nodes
    graph = build_agentic_graph()
    
    # Test query
    question = "What is the latest treatment for stage II melanoma?"
    
    print(f"\nQuestion: {question}")
    print("-"*60)
    
    # Run synchronously
    result = run_query_sync(graph, question)
    
    print(f"\nResults:")
    print(f"  Plan: {result.get('plan', [])}")
    print(f"  Final Answer: {result.get('final_answer', 'N/A')[:100]}...")
    print(f"  Confidence: {result.get('final_confidence', 0)}/10")
    print(f"  Iterations: {result.get('iteration_count', 0)}")
    print(f"  Reasoning Steps: {len(result.get('reasoning_trace', []))}")
    
    return result


def test_retriever_only():
    """Test just the retriever agent with PubMed."""
    print("\n" + "="*60)
    print("TEST 2: Retriever Agent (PubMed search)")
    print("="*60)
    
    from src.agents.retriever import RetrieverAgent
    from src.orchestrator.state import create_initial_state
    
    # Create agent
    retriever = RetrieverAgent(max_results=5)
    
    # Create state with PICO query
    state = create_initial_state("What is the treatment for melanoma?")
    state["pico_query"] = {
        "population": ["melanoma", "cutaneous melanoma"],
        "intervention": ["treatment", "therapy"],
        "modifiers": ["stage II"],
    }
    
    print(f"\nPICO Query: {state['pico_query']}")
    print("-"*60)
    
    # Run retrieval (sync)
    import asyncio
    result = asyncio.run(retriever.retrieve(state))
    
    docs = result.get("retrieved_docs", [])
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs[:3]):
        print(f"  {i+1}. PMID:{doc['pmid']} - {doc['title'][:60]}...")
    
    return result


def test_full_pipeline():
    """Test the full agentic pipeline with real agents."""
    print("\n" + "="*60)
    print("TEST 3: Full Agentic Pipeline")
    print("="*60)
    
    try:
        from src.orchestrator.graph import build_full_agentic_graph, run_query_sync
        
        # Build graph with real agents
        print("\nBuilding graph with real agents...")
        graph = build_full_agentic_graph()
        
        # Test query
        question = "What are the survival rates for stage II melanoma treatment?"
        
        print(f"\nQuestion: {question}")
        print("-"*60)
        
        # Run query
        print("\nRunning query (this may take 30-60 seconds)...")
        result = run_query_sync(graph, question, max_iterations=2)
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print("="*60)
        print(f"\nPlan Steps:")
        for i, step in enumerate(result.get('plan', [])[:5]):
            print(f"  {i+1}. {step}")
        
        print(f"\nRetrieved Docs: {len(result.get('retrieved_docs', []))}")
        print(f"Extracted Notes: {len(result.get('extracted_notes', []))}")
        
        print(f"\nFinal Answer:")
        print("-"*40)
        answer = result.get('final_answer', 'No answer generated')
        print(answer[:500] + "..." if len(answer) > 500 else answer)
        
        print(f"\nConfidence: {result.get('final_confidence', 0)}/10")
        print(f"Iterations: {result.get('iteration_count', 0)}")
        
        print(f"\nReasoning Trace:")
        for step in result.get('reasoning_trace', [])[:5]:
            print(f"  [{step.get('phase')}] {step.get('thought', '')[:60]}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🧪 Agentic RAG Test Suite")
    print("="*60)
    
    # Test 1: Placeholder (always works)
    test_with_placeholder_graph()
    
    # Test 2: Just retriever
    try:
        test_retriever_only()
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
    
    # Test 3: Full pipeline (requires LLMs)
    print("\n" + "="*60)
    print("Note: Test 3 requires Ollama running or Gemini API key")
    print("="*60)
    
    run_full_test = input("\nRun full pipeline test? (y/n): ").strip().lower()
    if run_full_test == 'y':
        test_full_pipeline()
    else:
        print("Skipped full pipeline test.")
    
    print("\n✅ Tests complete!")
