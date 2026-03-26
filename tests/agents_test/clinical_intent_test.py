# test_clinical_intent.py
import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.clinical_intent import ClinicalIntentAgent
from src.core.registry import ModelRegistry
from src.state.state import GraphState
import logging

# Set up logging to see the output
logging.basicConfig(level=logging.INFO)

async def test_single_question(question: str):
    """Test the ClinicalIntentAgent with a single question"""
    
    print(f"\n{'='*60}")
    print(f"Testing: '{question}'")
    print('='*60)
    
    # Initialize the agent
    agent = ClinicalIntentAgent()
    
    # Create a minimal state
    state: GraphState = {
        "original_question": question,
        "intent": None,
        "risk_level": None,
        "requires_disclaimer": None,
        "needs_guidelines": None,
        "confidence": None,
        "reasoning": None,
        "safety_flags": [],
        "messages": [],
        "documents": [],
        "generation": None,
        "rewritten_question": None,
        "conversation_id": "test-123",
        "processing_time": None
    }
    
    try:
        # Run the classification
        result = await agent.classify(state)
        
        # Print results nicely
        print("\nCLASSIFICATION RESULTS:")
        print(f"  Intent: {result.get('intent', 'N/A')}")
        print(f"  Risk Level: {result.get('risk_level', 'N/A')}")
        print(f"  Requires Disclaimer: {result.get('requires_disclaimer', 'N/A')}")
        print(f"  Needs Guidelines: {result.get('needs_guidelines', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A'):.2f}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
        
        if result.get('safety_flags'):
            print(f"  Safety Flags: {result['safety_flags']}")
            
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

async def run_multiple_tests():
    """Test multiple questions to see different classifications"""
    
    test_questions = [
        # Informational questions
        "What is diabetes?",
        "How does insulin work in the body?",
        
        # Clinical/Personal questions
        "Can I stop taking my blood pressure medication if I feel fine?",
        "Should I take aspirin for my chest pain?",
        
        # Emergency symptoms
        "I'm having chest pain and shortness of breath right now",
        "My father is unconscious and not breathing",
        
        # Edge cases
        "",  # Empty question
        "What is the meaning of life?",  # Non-medical
        "I need help with my depression medication dosage"  # Mixed
    ]
    
    for question in test_questions:
        await test_single_question(question)
        await asyncio.sleep(1)  # Small delay between tests
        print("\n" + "-"*60 + "\n")

async def test_with_custom_question():
    """Interactive mode - type your own questions"""
    
    print("Clinical Intent Classifier Test Harness")
    print("Type 'quit' to exit, or press Enter for default tests\n")
    
    agent = ClinicalIntentAgent()
    
    while True:
        question = input("\nEnter a medical question: ").strip()
        
        if question.lower() == 'quit':
            break
            
        if not question:
            # Run default tests
            await run_multiple_tests()
            continue
        
        # Create state and test
        state: GraphState = {
            "original_question": question,
            "intent": None,
            "risk_level": None,
            "requires_disclaimer": None,
            "needs_guidelines": None,
            "confidence": None,
            "reasoning": None,
            "safety_flags": [],
            "messages": [],
            "documents": [],
            "generation": None,
            "rewritten_question": None,
            "conversation_id": "interactive-test",
            "processing_time": None
        }
        
        result = await agent.classify(state)
        
        # Print in a more readable format
        print(f"\n[OK] Intent: {result.get('intent')}")
        print(f"[!]  Risk: {result.get('risk_level')}")
        print(f"[?] Reasoning: {result.get('reasoning')}")

if __name__ == "__main__":
    # Check if a question was provided as command line argument
    if len(sys.argv) > 1:
        # Run with command line argument
        question = " ".join(sys.argv[1:])
        asyncio.run(test_single_question(question))
    else:
        # Run interactive mode
        asyncio.run(test_with_custom_question())