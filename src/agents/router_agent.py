import logging
from typing import Dict, Any, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.core.registry import ModelRegistry
from src.state.state import GraphState, RouterOutput

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are the Master Router for a medical RAG system.
Your job is to optimally route queries to the most efficient execution path based on complexity and ambiguity.

INPUTS:
1. Query: The user's medical question.
2. Clinical Context:
   - Intent: {intent} (therapeutic/diagnostic/mechanism/informational)
   - Risk Level: {risk_level} (high/medium/low)

MODES:
1. "direct_qa" (Fast Path):
   - For simple, unambiguous questions.
   - Examples: "What is diabetes?", "Side effects of aspirin?", "Is X a drug?"
   - Criteria: Single-hop, definitional, or general knowledge.

2. "disambiguation" (Medium Path):
   - For specific but potentially ambiguous questions requiring clean context.
   - Examples: "Effect of Drug X on Gene Y?", "Does study Z support this?"
   - Criteria: Single-hop but requires precise retrieval from specific literature; high risk of noise.

3. "multihop" (Slow Path):
   - For complex queries requiring reasoning across multiple steps/documents.
   - Examples: "Compare X and Y treatments", "How does X cause Y via Z?", "Best management for patient with A and B?"
   - Criteria: Multi-step, comparative, or patient-specific management.

OUTPUT FORMAT (JSON):
{{
    "execution_mode": "direct_qa" | "disambiguation" | "multihop",
    "requires_planning": bool,         # True if mode is multihop
    "requires_extraction": bool,       # True if mode is disambiguation or multihop
    "requires_evidence_grading": bool, # True if mode is disambiguation or multihop
    "answer_policy": {{
        "format": "standard" | "yes_no",   # Use "yes_no" if query is binary
        "force_commitment": bool,          # True if query is binary (Yes/No)
        "allow_disclaimer": bool           # False if mode is direct_qa (strict), True otherwise
    }}
}}
"""

ROUTER_HUMAN_PROMPT = """Query: {question}"""

class RouterAgent:
    """
    Agent responsible for routing queries to the correct execution pipeline.
    """
    def __init__(self):
        # Use a fast but capable model (Flash/Heavy)
        # Using Heavy for reliability in routing logic as per spec
        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", ROUTER_HUMAN_PROMPT)
        ])
    
    async def route(self, state: GraphState) -> Dict[str, Any]:
        """
        Analyze the query and determine the execution mode.
        """
        question = state["original_question"]
        intent = state.get("intent", "informational")
        risk_level = state.get("risk_level", "low")
        
        logger.info(f"Routing query: {question[:50]}... [Intent: {intent}, Risk: {risk_level}]")
        
        try:
            chain = self.prompt | self.llm | self.parser
            response = await chain.ainvoke({
                "question": question,
                "intent": intent,
                "risk_level": risk_level
            })
            
            # Helper to ensure type safety for Literal
            mode = response.get("execution_mode", "multihop")
            if mode not in ["direct_qa", "disambiguation", "multihop"]:
                logger.warning(f"Invalid mode '{mode}' returned LLM. Defaulting to 'multihop'.")
                mode = "multihop"
                
            router_output: RouterOutput = {
                "execution_mode": mode,
                "requires_planning": response.get("requires_planning", True),
                "requires_extraction": response.get("requires_extraction", True),
                "requires_evidence_grading": response.get("requires_evidence_grading", True),
                "answer_policy": response.get("answer_policy", {
                    "format": "standard", 
                    "force_commitment": False,
                    "allow_disclaimer": True
                }),
                "execution_budget": None # Placeholder for future enforcement
            }
            
            logger.info(f"Router Decision: {mode}")
            return {"router_output": router_output}
            
        except Exception as e:
            logger.error(f"Router failed: {e}. Fallback to multihop.")
            # Safe Fallback to existing Complex Path
            fallback_output: RouterOutput = {
                "execution_mode": "multihop",
                "requires_planning": True,
                "requires_extraction": True,
                "requires_evidence_grading": True,
                "answer_policy": {"format": "standard", "force_commitment": False, "allow_disclaimer": True},
                "execution_budget": None
            }
            return {"router_output": fallback_output}

async def router_node(state: GraphState) -> Dict[str, Any]:
    agent = RouterAgent()
    return await agent.route(state)
