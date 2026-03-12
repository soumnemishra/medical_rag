
import logging
import asyncio
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.planner import PlannerAgent
    from src.agents.clinical_intent import ClinicalIntentAgent
    from src.agents.safety_critic import ClinicalSafetyCriticAgent
    from src.agents.step_definer import StepDefinerAgent
    from src.agents.rag import RagAgent
    from src.agents.extractor import ExtractorAgent
    from src.agents.evidence_polarity_agent import EvidencePolarityAgent

logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    Singleton registry for managing persistent agent instances.
    Ensures that each agent is instantiated exactly once across the application lifecycle.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        if AgentRegistry._instance is not None:
             raise RuntimeError("AgentRegistry is a singleton. Use get_instance() instead.")
        
        self._planner: Optional[PlannerAgent] = None
        self._clinical_intent: Optional[ClinicalIntentAgent] = None
        self._safety_critic: Optional[ClinicalSafetyCriticAgent] = None
        self._step_definer: Optional[StepDefinerAgent] = None
        self._rag: Optional[RagAgent] = None
        self._extractor: Optional[ExtractorAgent] = None
        self._evidence_polarity: Optional[EvidencePolarityAgent] = None
        
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """Get the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """
        Eagerly initialize all agents. 
        This should be called at application startup.
        """
        if self._initialized:
            logger.info("AgentRegistry already initialized.")
            return

        logger.info("Initializing AgentRegistry and all Agents...")
        
        # Instantiate agents
        from src.agents.planner import PlannerAgent
        from src.agents.clinical_intent import ClinicalIntentAgent
        from src.agents.safety_critic import ClinicalSafetyCriticAgent
        from src.agents.step_definer import StepDefinerAgent
        from src.agents.rag import RagAgent
        from src.agents.extractor import ExtractorAgent
        from src.agents.evidence_polarity_agent import EvidencePolarityAgent

        self._planner = PlannerAgent()
        self._clinical_intent = ClinicalIntentAgent()
        self._safety_critic = ClinicalSafetyCriticAgent()
        self._step_definer = StepDefinerAgent()
        self._rag = RagAgent()
        self._extractor = ExtractorAgent()
        self._evidence_polarity = EvidencePolarityAgent()
        
        self._initialized = True
        logger.info("AgentRegistry initialization complete.")

    @property
    def planner(self) -> "PlannerAgent":
        if not self._planner:
            raise RuntimeError("Registry not initialized. Call initialize() first.")
        return self._planner

    @property
    def clinical_intent(self) -> "ClinicalIntentAgent":
        if not self._clinical_intent:
            raise RuntimeError("Registry not initialized. Call initialize() first.")
        return self._clinical_intent

    @property
    def safety_critic(self) -> "ClinicalSafetyCriticAgent":
        if not self._safety_critic:
             raise RuntimeError("Registry not initialized. Call initialize() first.")
        return self._safety_critic

    @property
    def step_definer(self) -> "StepDefinerAgent":
        if not self._step_definer:
            raise RuntimeError("Registry not initialized. Call initialize() first.")
        return self._step_definer
    
    @property
    def rag(self) -> "RagAgent":
        if not self._rag:
            raise RuntimeError("Registry not initialized. Call initialize() first.")
        return self._rag

    @property
    def extractor(self) -> "ExtractorAgent":
        if not self._extractor:
            raise RuntimeError("Registry not initialized. Call initialize() first.")
        return self._extractor

    @property
    def evidence_polarity(self) -> "EvidencePolarityAgent":
        if not self._evidence_polarity:
            raise RuntimeError("Registry not initialized. Call initialize() first.")
        return self._evidence_polarity

