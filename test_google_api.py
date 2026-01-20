import os
import sys
from langchain_core.messages import HumanMessage
from src.core.registry import ModelRegistry
from src.config import settings

def test_google_api():
    print(f"Checking Google API Key configuration...")
    
    if not settings.GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY is not set in environment or .env file.")
        return

    print(f"Key found: {settings.GOOGLE_API_KEY[:5]}...{settings.GOOGLE_API_KEY[-4:] if settings.GOOGLE_API_KEY else ''}")
    print(f"Testing Gemini Model: {settings.GEMINI_MODEL}")

    try:
        # Force Hybrid to ensured we get the Heavy LLM (Gemini)
        settings.USE_HYBRID = True 
        llm = ModelRegistry.get_heavy_llm()
        
        print("Model initialized. Sending request...")
        response = llm.invoke([HumanMessage(content="Hello, are you active? Reply with 'Yes, I am active.'")])
        
        print("\nSUCCESS! Google API is working.")
        print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"\nFAILURE: Google API check failed.")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_google_api()
