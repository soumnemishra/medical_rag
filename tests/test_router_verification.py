
import asyncio
import logging
from unittest.mock import MagicMock
from typing import Dict, Any

import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mocking the dependencies to avoid actual LLM calls
from src.state.state import GraphState, RouterOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock Responses for different scenarios
MOCK_RESPONSES = {
    "direct_qa": {
        "execution_mode": "direct_qa",
        "requires_planning": False,
        "requires_extraction": False,
        "requires_evidence_grading": False,
        "answer_policy": {"format": "yes_no", "force_commitment": True, "allow_disclaimer": False}
    },
    "disambiguation": {
        "execution_mode": "disambiguation",
        "requires_planning": False, 
        "requires_extraction": True,
        "requires_evidence_grading": True,
        "answer_policy": {"format": "standard", "force_commitment": False, "allow_disclaimer": True}
    },
    "multihop": {
        "execution_mode": "multihop",
        "requires_planning": True,
        "requires_extraction": True, 
        "requires_evidence_grading": True,
        "answer_policy": {"format": "standard", "force_commitment": False, "allow_disclaimer": True}
    }
}

async def test_router_logic():
    print("\n🧪 Starting Router Verification Tests (Mocked LLM)...\n")
    
    # 1. Import RouterAgent locally to patch it
    from src.agents import router_agent 
    
    # Patch ModelRegistry BEFORE instantiation to prevent real LLM init
    router_agent.ModelRegistry = MagicMock()
    router_agent.ModelRegistry.get_heavy_llm = MagicMock(return_value=MagicMock())
    
    # 2. Patch the LLM chain
    from src.agents.router_agent import RouterAgent
    agent = RouterAgent()
    
    # Mock the chain.ainvoke to return specific responses based on input query
    async def mock_invoke(input_dict: Dict[str, Any]):
        question = input_dict.get("question", "").lower()
        
        if "diabetes" in question or "insulin" in question:
            # Case 1: Simple/Binary -> Direct QA
            return MOCK_RESPONSES["direct_qa"]
        elif "effect of" in question or "aspirin" in question:
            # Case 2: Ambiguous -> Disambiguation
            return MOCK_RESPONSES["disambiguation"]
        else:
            # Case 3: Complex -> Multihop
            return MOCK_RESPONSES["multihop"]
            
    agent.prompt = MagicMock()
    agent.llm = MagicMock()
    agent.parser = MagicMock()
    
    # Create a mock chain object
    mock_chain = MagicMock()
    mock_chain.ainvoke = mock_invoke
    
    # Override the chain construction in the agent requires knowing how it's built or just mocking the whole pipe
    # In router_agent.py: chain = self.prompt | self.llm | self.parser
    # We can't easily patch the pipe operator result on the fly without changing code or careful mocking.
    # EASIER: Subclass RouterAgent or patch the route method? 
    # Let's Patch the 'chain' creation logic or just mock the components so the pipe returns our mock_chain.
    # Actually, let's just monkeypatch the `route` method's internal chain usage? No, that's hard.
    # Let's mock the `prompt | llm | parser` behavior. 
    # or simpler: Just mock agent.llm to return the JSON directly? 
    # The agent uses: chain = self.prompt | self.llm | self.parser
    # If we replace `self.llm` with a mock that returns the Dict, and `self.parser` with a mock that just returns output...
    
    # Let's try a simpler approach: Monkeypatch the `ainvoke` of the chain.
    # But we don't have access to the chain object until runtime.
    
    # STRATEGY: Patch `RouterAgent.route` slightly or rely on `ModelRegistry` mocking?
    # Let's go with: Mocking the `ainvoke` call by replacing the chain construction.
    
    # Actually, simpler: We can just mock the `agent.prompt`, `agent.llm` etc?
    # No, the pipe `|` creates a `RunnableSequence`.
    
    # Hacky but effective: Override `agent.prompt` with something that when piped returns our mock chain.
    class MockRunnable:
        def __or__(self, other):
            return self
        async def ainvoke(self, input_dm):
            return await mock_invoke(input_dm)
            
    agent.prompt = MockRunnable()
    agent.llm = MockRunnable() 
    agent.parser = MockRunnable()
    
    # TEST CASE 1: Direct QA
    print("🔹 Test Case 1: 'What is diabetes?' (Expect: direct_qa)")
    state_1: GraphState = {
        "original_question": "What is diabetes?",
        "intent": "informational",
        "risk_level": "low"
    }
    result_1 = await agent.route(state_1)
    output_1 = result_1["router_output"]
    
    assert output_1["execution_mode"] == "direct_qa"
    assert output_1["answer_policy"]["force_commitment"] is True
    assert output_1["answer_policy"]["allow_disclaimer"] is False
    print("✅ Passed: Output matched Direct QA schema.")
    
    # TEST CASE 2: Disambiguation
    print("\n🔹 Test Case 2: 'Effect of aspirin on X?' (Expect: disambiguation)")
    state_2: GraphState = {
        "original_question": "Effect of aspirin on blood?",
        "intent": "mechanism", 
        "risk_level": "medium"
    }
    result_2 = await agent.route(state_2)
    output_2 = result_2["router_output"]
    
    assert output_2["execution_mode"] == "disambiguation"
    assert output_2["requires_extraction"] is True
    print("✅ Passed: Output matched Disambiguation schema.")
    
    # TEST CASE 3: Multihop (Fallback)
    print("\n🔹 Test Case 3: 'Compare X and Y' (Expect: multihop)")
    state_3: GraphState = {
        "original_question": "Compare X and Y",
        "intent": "therapeutic",
        "risk_level": "high"
    }
    result_3 = await agent.route(state_3)
    output_3 = result_3["router_output"]
    
    assert output_3["execution_mode"] == "multihop"
    assert output_3["requires_planning"] is True
    print("✅ Passed: Output matched Multihop schema.")
    
    print("\n🎉 All Router Verification Tests Passed!")

if __name__ == "__main__":
    asyncio.run(test_router_logic())
