import logging
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.planner import PlannerAgent
    from src.agents.clinical_intent import ClinicalIntentAgent
    from src.agents.safety_critic import ClinicalSafetyCriticAgent
    from src.agents.step_definer import StepDefinerAgent
    from src.agents.rag import RagAgent
    from src.agents.extractor import ExtractorAgent
    from src.agents.evidence_polarity_agent import EvidencePolarityAgent
    from src.agents.evidence_decision_agent import EvidenceDecisionAgent
    from src.agents.router_agent import RouterAgent
    from src.tools.retriever import RetrieverTool

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Singleton registry for all agent and tool instances.

    Guarantees each agent and tool is constructed exactly once per process
    lifetime. This matters because:
      - RetrieverTool loads 50–100MB sentence-transformer models on init
      - LLM clients open connection pools on init
      - Building LCEL chains has non-trivial overhead

    Usage:
        # At app startup (app.py or graph.py):
        AgentRegistry.get_instance().initialize()

        # In any node or agent:
        agent = AgentRegistry.get_instance().planner

    Thread safety:
        get_instance() uses a threading.Lock with double-checked locking
        so it is safe to call from multiple async coroutines simultaneously.
        asyncio.Lock() is NOT used here because get_instance() is synchronous
        and asyncio locks cannot be acquired outside a running event loop.
    """

    _instance: Optional["AgentRegistry"] = None
    _lock = threading.Lock()   # threading.Lock — works in both sync and async

    def __init__(self):
        if AgentRegistry._instance is not None:
            raise RuntimeError("AgentRegistry is a singleton. Use get_instance().")

        # Tools
        self._retriever: Optional["RetrieverTool"]         = None

        # Agents
        self._planner:           Optional["PlannerAgent"]              = None
        self._clinical_intent:   Optional["ClinicalIntentAgent"]       = None
        self._safety_critic:     Optional["ClinicalSafetyCriticAgent"] = None
        self._step_definer:      Optional["StepDefinerAgent"]          = None
        self._rag:               Optional["RagAgent"]                  = None
        self._extractor:         Optional["ExtractorAgent"]            = None
        self._evidence_polarity: Optional["EvidencePolarityAgent"]     = None
        self._evidence_decision: Optional["EvidenceDecisionAgent"]     = None
        self._router:            Optional["RouterAgent"]               = None

        self._initialized = False

    # ---------------------------------------------------------------- #
    #  Singleton access                                                  #
    # ---------------------------------------------------------------- #

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """
        Return the singleton instance, creating it if necessary.
        Thread-safe via double-checked locking with threading.Lock.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:   # re-check after acquiring lock
                    cls._instance = cls()
        return cls._instance

    # ---------------------------------------------------------------- #
    #  Initialisation                                                    #
    # ---------------------------------------------------------------- #

    def initialize(self) -> None:
        """
        Eagerly initialise all agents and tools.
        Call this ONCE at application startup (in app.py or graph.py).

        Startup order matters:
          1. RetrieverTool first — agents that need it receive the singleton
          2. RagAgent receives the retriever via DI — no second model load
          3. All other agents load their LLMs independently

        If any agent raises RuntimeError (LLM unavailable), the whole
        startup fails loudly rather than silently degrading mid-query.
        """
        if self._initialized:
            logger.info("AgentRegistry already initialised — skipping.")
            return

        logger.info("Initialising AgentRegistry...")

        # ── Tools first ────────────────────────────────────────────
        from src.tools.retriever import RetrieverTool
        logger.info("  Loading RetrieverTool (sentence-transformer models)...")
        self._retriever = RetrieverTool()

        # ── Agents ─────────────────────────────────────────────────
        from src.agents.planner               import PlannerAgent
        from src.agents.clinical_intent       import ClinicalIntentAgent
        from src.agents.safety_critic         import ClinicalSafetyCriticAgent
        from src.agents.step_definer          import StepDefinerAgent
        from src.agents.rag                   import RagAgent
        from src.agents.extractor             import ExtractorAgent
        from src.agents.evidence_polarity_agent  import EvidencePolarityAgent
        from src.agents.evidence_decision_agent  import EvidenceDecisionAgent
        from src.agents.router_agent          import RouterAgent

        logger.info("  Initialising agents...")
        self._planner          = PlannerAgent()
        self._clinical_intent  = ClinicalIntentAgent()
        self._safety_critic    = ClinicalSafetyCriticAgent()
        self._step_definer     = StepDefinerAgent()

        # RagAgent receives the shared retriever — no second model load
        self._rag              = RagAgent(retriever_tool=self._retriever)

        self._extractor        = ExtractorAgent()
        self._evidence_polarity = EvidencePolarityAgent()
        self._evidence_decision = EvidenceDecisionAgent()
        self._router           = RouterAgent()

        self._initialized = True
        logger.info("AgentRegistry initialisation complete.")

    def _lazy_init(self, attr: str, factory) -> object:
        """
        Lazy-init fallback: if initialize() was not called at startup,
        individual agents are created on first access rather than crashing.
        This makes forgetting initialize() a performance issue, not a crash.
        """
        val = getattr(self, attr)
        if val is None:
            logger.warning(
                f"AgentRegistry: '{attr}' accessed before initialize() — "
                f"lazy-initialising. Call initialize() at startup to avoid this."
            )
            val = factory()
            setattr(self, attr, val)
        return val

    # ---------------------------------------------------------------- #
    #  Tool properties                                                   #
    # ---------------------------------------------------------------- #

    @property
    def retriever(self) -> "RetrieverTool":
        from src.tools.retriever import RetrieverTool
        return self._lazy_init("_retriever", RetrieverTool)

    # ---------------------------------------------------------------- #
    #  Agent properties                                                  #
    # ---------------------------------------------------------------- #

    @property
    def planner(self) -> "PlannerAgent":
        from src.agents.planner import PlannerAgent
        return self._lazy_init("_planner", PlannerAgent)

    @property
    def clinical_intent(self) -> "ClinicalIntentAgent":
        from src.agents.clinical_intent import ClinicalIntentAgent
        return self._lazy_init("_clinical_intent", ClinicalIntentAgent)

    @property
    def safety_critic(self) -> "ClinicalSafetyCriticAgent":
        from src.agents.safety_critic import ClinicalSafetyCriticAgent
        return self._lazy_init("_safety_critic", ClinicalSafetyCriticAgent)

    @property
    def step_definer(self) -> "StepDefinerAgent":
        from src.agents.step_definer import StepDefinerAgent
        return self._lazy_init("_step_definer", StepDefinerAgent)

    @property
    def rag(self) -> "RagAgent":
        from src.agents.rag import RagAgent
        # Inject shared retriever so sentence-transformer models load once
        return self._lazy_init(
            "_rag",
            lambda: RagAgent(retriever_tool=self.retriever)
        )

    @property
    def extractor(self) -> "ExtractorAgent":
        from src.agents.extractor import ExtractorAgent
        return self._lazy_init("_extractor", ExtractorAgent)

    @property
    def evidence_polarity(self) -> "EvidencePolarityAgent":
        from src.agents.evidence_polarity_agent import EvidencePolarityAgent
        return self._lazy_init("_evidence_polarity", EvidencePolarityAgent)

    @property
    def evidence_decision(self) -> "EvidenceDecisionAgent":
        from src.agents.evidence_decision_agent import EvidenceDecisionAgent
        return self._lazy_init("_evidence_decision", EvidenceDecisionAgent)

    @property
    def router(self) -> "RouterAgent":
        from src.agents.router_agent import RouterAgent
        return self._lazy_init("_router", RouterAgent)