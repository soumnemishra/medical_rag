import logging
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

from src.state.state import RagState, QAAnswerFormat
from src.prompts.templates import QA_SYSTEM_PROMPT, QA_HUMAN_PROMPT
from src.core.registry import ModelRegistry
from src.tools.retriever import RetrieverTool

logger = logging.getLogger(__name__)


class RagAgent:
    """
    Executes the final QA step for a single plan step.

    Pipeline:
      1. Use provided documents (from ExtractorAgent) OR retrieve fresh
      2. Format context string from documents
      3. Pass context + question + evidence_polarity to LLM
      4. Return structured answer with citations

    evidence_polarity is passed through to the prompt so the LLM knows
    whether to present a balanced view (mixed/refute) or a confident
    answer (support). This is the connection between EvidencePolarityAgent
    and the final answer quality.

    Dependency injection:
        __init__ accepts an optional retriever_tool so tests can inject
        a mock without loading real sentence-transformer models.
        In production, the registry's singleton retriever is passed in.
    """

    def __init__(self, retriever_tool: RetrieverTool = None):
        # Accept injected retriever (from registry) or create one.
        # Production callers should always inject the singleton from
        # AgentRegistry to avoid reloading sentence-transformer models.
        self.retriever = retriever_tool

        self.llm = ModelRegistry.get_heavy_llm(temperature=0.0, json_mode=True)

        if self.llm is None:
            raise RuntimeError(
                "RagAgent: Heavy LLM failed to load. "
                "Check Ollama is running or GOOGLE_API_KEY is set."
            )

        # QAAnswerFormat was imported but never wired in — fixed here.
        # Bare JsonOutputParser() silently returns {} when the LLM uses
        # "answer_text" instead of "answer". This schema makes it throw.
        self.parser = JsonOutputParser(pydantic_object=QAAnswerFormat)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            ("human",  QA_HUMAN_PROMPT),
        ])

        # Build chain ONCE — reused across all query() calls
        self.chain = self.prompt | self.llm | self.parser

    # ---------------------------------------------------------------- #
    #  Private helpers                                                   #
    # ---------------------------------------------------------------- #

    def _get_retriever(self) -> RetrieverTool:
        """
        Lazy-load retriever only when actually needed (no pre-provided docs).
        Avoids loading sentence-transformer models for calls that already
        have documents from ExtractorAgent.
        """
        if self.retriever is None:
            logger.info("RagAgent: lazy-loading RetrieverTool (no injected retriever)")
            self.retriever = RetrieverTool()
        return self.retriever

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def _invoke_chain(self, inputs: dict) -> dict:
        """
        Runs the QA chain with up to 2 retries on transient failures.
        Isolated for independent testability.
        """
        return await self.chain.ainvoke(inputs)

    def _extract_polarity_string(self, state: RagState) -> str:
        """
        Safely extract a polarity string from state for prompt injection.

        QA_HUMAN_PROMPT has {evidence_polarity} placeholder. This must
        always be a string — never None or a dict — or LangChain raises
        a template formatting error.
        """
        polarity_data = state.get("evidence_polarity", {})
        if isinstance(polarity_data, dict):
            return polarity_data.get("polarity", "insufficient")
        if isinstance(polarity_data, str):
            return polarity_data
        return "insufficient"

    # ---------------------------------------------------------------- #
    #  Public interface                                                  #
    # ---------------------------------------------------------------- #

    async def query(self, state: RagState) -> Dict[str, Any]:
        """
        Execute RAG QA for a single step question.

        Reads from RagState:
            question           — the step question to answer
            documents          — pre-extracted notes (from ExtractorAgent)
                                 if provided, retrieval is skipped
            doc_ids            — PMIDs for the provided documents
            evidence_polarity  — polarity classification for prompt context

        Returns:
            question           — echoed back for traceability
            documents          — the contexts used (retrieved or provided)
            doc_ids            — PMIDs used
            notes              — analysis text for memory/aggregation
            final_raw_answer   — full LLM response dict
        """
        question         = state["question"]
        polarity_str     = self._extract_polarity_string(state)

        # ── 1. Get documents ───────────────────────────────────────
        if state.get("documents"):
            # ExtractorAgent already filtered and graded the docs —
            # use them directly, skip retrieval
            contexts = state["documents"]
            doc_ids  = state.get("doc_ids", [])
            logger.info(
                f"RagAgent: using {len(contexts)} provided document(s), "
                f"skipping retrieval"
            )
        else:
            # No pre-extracted docs — retrieve from PubMed now
            logger.info(f"RagAgent: retrieving documents for '{question[:60]}...'")
            try:
                contexts, doc_ids = await self._get_retriever()(question)
            except Exception as e:
                logger.error(f"RagAgent retrieval failed: {e}", exc_info=True)
                contexts, doc_ids = [], []

        # ── 2. Build context string ────────────────────────────────
        if not contexts:
            context_text = "No relevant documents found."
            logger.warning(f"RagAgent: zero documents for question='{question[:60]}...'")
        else:
            context_text = "\n\n".join(str(c) for c in contexts)

        # ── 3. Generate answer ─────────────────────────────────────
        try:
            try:
                response = await self._invoke_chain({
                    "context":           context_text,
                    "question":          question,
                    "evidence_polarity": polarity_str,  # feeds {evidence_polarity} in prompt
                })
            except RetryError as e:
                raise ValueError(f"QA chain failed after 2 attempts: {e}") from e

            analysis = response.get("analysis", "No analysis provided")

            logger.info(
                f"RagAgent answer: '{response.get('answer', '')[:120]}...' "
                f"| polarity={polarity_str}"
            )

            return {
                "question":        question,
                "documents":       contexts,
                "doc_ids":         doc_ids,
                "notes":           [analysis],
                "final_raw_answer": response,
            }

        except Exception as e:
            logger.error(f"RagAgent generation failed: {e}", exc_info=True)
            fallback = {
                "analysis": f"Generation error: {e}",
                "answer":   "Failed to generate answer due to an error.",
                "success":  "No",
                "rating":   0,
                "is_error": True,
            }
            return {
                "question":         question,
                "documents":        [],
                "doc_ids":          [],
                "notes":            ["Error during generation"],
                "final_raw_answer": fallback,
            }


# ------------------------------------------------------------------ #
#  LangGraph node wrapper                                             #
# ------------------------------------------------------------------ #

from src.agents.registry import AgentRegistry


async def rag_node(state: RagState) -> Dict[str, Any]:
    """
    Thin wrapper called by LangGraph StateGraph.
    Uses registry singleton so the retriever and LLM are not reloaded
    per call.
    """
    try:
        agent = AgentRegistry.get_instance().rag
        return await agent.query(state)
    except Exception as e:
        logger.error(f"rag_node crashed: {e}", exc_info=True)
        return {
            "question":         state.get("question", ""),
            "documents":        [],
            "doc_ids":          [],
            "notes":            [],
            "final_raw_answer": {
                "answer":   "Node-level error during RAG execution.",
                "success":  "No",
                "rating":   0,
                "is_error": True,
            },
        }