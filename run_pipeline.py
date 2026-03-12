import asyncio
import logging
from src.config import settings, configure_logging
from src.orchestrator.graph import build_graph

# Setup logging
configure_logging()
logger = logging.getLogger(__name__)

async def main():
    print("Initializing MA-RAG Pipeline...")
    from src.agents.registry import AgentRegistry
    AgentRegistry.get_instance().initialize()
    
    graph = build_graph()
    
    question = "What are the effective treatments for Hirschsprung's Disease?"
    print(f"\nProcessing Query: {question}\n")
    
    initial_state = {
        "original_question": question,
        "past_exp": [],
        "plan": [],
        "final_answer": ""
    }
    
    try:
        # Using ainvoke for async execution
        result = await graph.ainvoke(initial_state)
        
        print("\n" + "="*50)
        print("FINAL RESULT")
        print("="*50)
        print(f"Answer: {result.get('final_answer')}")
        
        print("\nExecution History:")
        for i, exp in enumerate(result.get("past_exp", [])):
            summary = exp.get("plan_summary", {})
            print(f"\nTrial {i+1}:")
            print(f"  Status: {summary.get('output')}")
            print(f"  Score: {summary.get('score')}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
