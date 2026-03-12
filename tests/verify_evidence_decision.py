
import asyncio
import logging
import unittest
from src.state.state import GraphState
from src.agents.evidence_decision_agent import EvidenceDecisionAgent, evidence_decision_node
from src.agents.supplemental_retrieval_node import supplemental_retrieval_node

# Mock retrieval tool to avoid hitting PubMed during unit test
class MockRetriever:
    async def __call__(self, query):
        if "evidence against" in query:
            return ["Evidence B refutes A", "Evidence C refutes A"], ["99999", "88888"]
        return ["Generic review"], ["12345"]

class TestEvidenceDecision(unittest.TestCase):
    def setUp(self):
        self.agent = EvidenceDecisionAgent()
        
    def test_overlap_calculation(self):
        past_exp = [
            {"step_docs_ids": [["1", "2"]]}, # List of lists format handling
            {"step_docs_ids": ["1", "3"]}
        ]
        # Current is ["1", "3"], Previous is ["1", "2"]. Intersection "1". Overlap 1/2 = 0.5
        overlap = self.agent._calculate_overlap(past_exp)
        self.assertEqual(overlap, 0.5)

        past_exp_high = [
            {"step_docs_ids": ["1", "2", "3"]},
            {"step_docs_ids": ["1", "2", "4"]}
        ]
        # Intersection "1", "2". Overlap 2/3 = 0.66
        overlap_high = self.agent._calculate_overlap(past_exp_high)
        self.assertGreater(overlap_high, 0.6)

async def manual_run():
    logging.basicConfig(level=logging.INFO)
    print("=== Test 1: Max Retry Logic ===")
    state = {
        "original_question": "Test",
        "evidence_polarity": {"polarity": "insufficient", "confidence": 0.0},
        "retry_count": 1,
        "past_exp": []
    }
    result = await evidence_decision_node(state)
    print(f"Result (Should be accept): {result}")

    print("\n=== Test 2: Supplemental Retrieval (Mocked if possible, else real) ===")
    # We won't mock the internal retriever tool here easily without injection, 
    # so we'll test the node logic constructs locally if possible, or run it and expect log output
    # For now, just verifying import and basic execution path
    try:
        from src.agents.supplemental_retrieval_node import supplemental_retrieval_node
        state_r = {
            "original_question": "Does aspirin cause bleeding?",
            "evidence_decision": "reretrieve_counter",
            "retry_count": 0,
            "past_exp": []
        }
        # This will hit real PubMed if not mocked, which is fine for integration test
        # result_r = await supplemental_retrieval_node(state_r)
        # print(f"Supplemental Result keys: {result_r.keys()}")
    except Exception as e:
        print(f"Skipping live retrieval test: {e}")

if __name__ == "__main__":
    asyncio.run(manual_run())
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
