# FILE: src/agent.py
"""
Pydantic AI agent for medical query answering with PubMed context.

Uses Vertex AI (Gemini) with Application Default Credentials for authentication.
Searches PubMed in real-time using PICO-based query decomposition.

Example Usage:
    from src.agent import MedicalAgent
    
    agent = MedicalAgent()
    response = await agent.answer_query(
        "What are the latest treatments for Type 2 diabetes?"
    )
    print(response.answer)
    print(response.sources)

Authentication:
    Run `gcloud auth application-default login` to set up credentials.
"""

import logging # logging up for the process 
import os
import asyncio
import random
import time
from dataclasses import dataclass, field #data class 
from typing import List, Optional


# basemodel verifies the structure of the data it is a pydantic 
#base model is the data validation system 
from pydantic import BaseModel, Field #base model is only used for the output classes not for shared 


#  agent wraps the llm and run context gives tools to access the shared memory 
from pydantic_ai import Agent, RunContext
from pydantic_ai import UsageLimits
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.ollama import OllamaProvider

from src.config import settings # config settings 
from src.pubmed_client import PubMedClient, RetrievedDocument  #pubmed client and retreiver 

# query builder 
from src.query_builder import (
    PubMedQueryBuilder,
    PICOQuery,
)
#error logging for the gemini api 
from src.exceptions import GeminiAPIError

logger = logging.getLogger(__name__)

# this is the model that will be used to generate the answer
# this defines what does valid medical answer should look like 
''' extends the base model with additional fields 
 study_quality_score etc 
 this strucutres the llm answers output validation 
 This is not LLM memory.
This is API/UI output.'''
class MedicalQueryResult(BaseModel):  #basemodel is given to the medical 
    #query result is the final output of the agent
    """
    Structured result from the medical agent.
    
    Attributes:
        answer: The AI-generated answer to the medical query.
        sources: List of PubMed article sources used.
        confidence: Confidence level (low/medium/high).
        query_log: List of built queries for display.
        reasoning_steps: Chain of thought reasoning steps.
        disclaimer: Medical disclaimer text.
    """
    # ai generated answer to the medical query based on the retrieved articles
    answer: str = Field(description="Evidence-based answer to the query")
    # list of pmids of the retrieved articles
    sources: List[str] = Field(default_factory=list, description="PMID references")
    # confidence level of the answer (low/medium/high)
    confidence: str = Field(default="medium", description="Confidence level")
    # list of built queries for display
    query_log: List[dict] = Field(default_factory=list, description="Built queries log")
    # chain of thought reasoning steps
    reasoning_steps: List[dict] = Field(default_factory=list, description="Chain of thought steps")
    provider_used: str = Field(default="unknown", description="Provider that produced the final answer")
    disclaimer: str = Field(
        default="This information is for educational purposes only. "
                "Always consult a healthcare professional for medical advice.",
        description="Medical disclaimer"
    )
########################################################################################################
################ this creates a shared################################################################## 
# memory for the agent to access and modify data during the conversation    
#The shared memory for ONE user query.

# Tools write to deps, Agent reads from / accesses deps
# MedicalAgentDeps ==> container that holds everything the AI 
# needs while working on ONE medical question.
# It acts as a shared storage for the agent to access 
# and modify data during the conversation.
@dataclass
class MedicalAgentDeps:
    """Dependencies injected into the agent.
    """
    pubmed_client: PubMedClient  # tools need to access pubmed client
    user_query: str = ""  # user query stored for context log and debugging 
    # default_factory=list==>default value is empty list acts as a storage for retrieved docs 
    retrieved_articles: List[RetrievedDocument] = field(default_factory=list)  # retrieved articles stored here 
    query_log: List[dict] = field(default_factory=list)  # Track built queries
    reasoning_log: List[dict] = field(default_factory=list)  # Track COT reasoning steps

# without the system prompt the llm hallucinates donot have the context and instructions for the conversations 
# System prompt for the medical agent with Chain of Thought
MEDICAL_SYSTEM_PROMPT = """You are a constrained medical retrieval engine.

Rules:
- Use search_pubmed_pico for every medical question before answering.
- Never answer from memory alone.
- Never reveal chain-of-thought, hidden reasoning, or internal deliberation.
- Never write filler such as "Here is the JSON you asked for", "Sure", or "Certainly".
- Tool arguments MUST BE RAW JSON. DO NOT wrap tool arguments in ```json ... ``` markdown blocks.
- If a PICO field is missing, use [] or null.
- Prefer short, factual, evidence-based answers with inline PMID citations.
- If the prompt is a benchmark question, end with exactly: The final answer is: [Letter].
- Do not add any text after the final answer line.
"""




# ########################################### BRAIN ######################################################
'''CREATED A CLASS PYTHON 
RESUABLE AROUND DIFFERENT QUERIES DONT NEED TO CONFIGURE 

one configured LLM

one configured tool set

one configured prompt

SUITABLE FOR CREATING INSTANCES '''

#This block builds the “AI brain” by selecting the model, 
# enforcing behavior, attaching memory, limiting output type, enabling tools, and finalizing setup
class MedicalAgent: #reusable object.
    """
    You don’t want to reconfigure Gemini, tools, and prompts for every query

    You want one agent instance handling many queries
    Pydantic AI agent for answering medical queries with PubMed context.
    
    Uses Vertex AI (Gemini) with Application Default Credentials.
    """
    
    def __init__(self, pubmed_client: Optional[PubMedClient] = None):
        """
        Initialize the medical agent with Vertex AI.
        
        Args:
            pubmed_client: Optional PubMed client instance.
        
        Requires:
            - GOOGLE_CLOUD_PROJECT set in environment
            - Application Default Credentials configured
              (run: gcloud auth application-default login)
        """
        # THIS REMVOES THE 
        '''ALLOWS FLEXIBLE INJECTION OF A PUB MED HANDLER ENABLING TESTABILITY AND SAFE RESULTS 
        
        JUST LIKE CREATING A ENGINE OR A DEFAULT ENGINE '''
        self.pubmed_client = pubmed_client or PubMedClient()

        self.query_builder = PubMedQueryBuilder()

        logger.info("Medical agent initialized, ready for cascades.")

    def close(self) -> None:
        """Release underlying client resources."""
        self.pubmed_client.close()

    def _build_usage_limits(self, query: str, use_tools: bool):
        """Choose a small tool-call budget based on query complexity."""
        if not use_tools:
            return None

        normalized_query = query.lower()
        complexity_score = 0

        if len(normalized_query) > 180:
            complexity_score += 1
        if len(normalized_query) > 350:
            complexity_score += 1
        if any(marker in normalized_query for marker in (" and ", " or ", " vs ", " compared ", " comparison ", " effect ", " impact ", " association ", " relationship ")):
            complexity_score += 1
        if any(symbol in normalized_query for symbol in (";", "/", "(", ")", ",")):
            complexity_score += 1

        tool_calls_limit = min(4, max(2, 2 + complexity_score))
        logger.debug(
            "Computed tool-call budget",
            extra={
                "tool_calls_limit": tool_calls_limit,
                "complexity_score": complexity_score,
                "query_preview": normalized_query[:120],
            },
        )
        return UsageLimits(tool_calls_limit=tool_calls_limit)

    def _get_model_instance(self, provider_name: str):
        pn = provider_name.lower().strip()
        if pn == "gemini":
            project = settings.GOOGLE_CLOUD_PROJECT
            location = settings.GOOGLE_CLOUD_LOCATION

            if settings.GOOGLE_API_KEY is None and project == "your-gcp-project-id":
                raise ValueError(
                    "Gemini unavailable: configure GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT for Vertex auth."
                )

            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project)
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)
            if settings.GOOGLE_API_KEY:
                return GeminiModel(
                    settings.GEMINI_MODEL,
                    provider=GoogleGLAProvider(api_key=settings.GOOGLE_API_KEY),
                )
            return GeminiModel(settings.GEMINI_MODEL, provider="google-vertex")
        elif pn == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("Groq unavailable: GROQ_API_KEY is not configured.")
            return GroqModel(
                settings.GROQ_MODEL,
                provider=GroqProvider(api_key=settings.GROQ_API_KEY),
            )
        elif pn == "github":
            if not settings.GITHUB_TOKEN:
                raise ValueError("GitHub Models unavailable: GITHUB_TOKEN is not configured.")
            return OpenAIModel(
                settings.GITHUB_MODEL,
                provider=OpenAIProvider(
                    base_url="https://models.inference.ai.azure.com",
                    api_key=settings.GITHUB_TOKEN,
                ),
            )
        elif pn == "ollama":
            ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")
            if ollama_base_url.endswith("/api"):
                ollama_base_url = f"{ollama_base_url[:-4]}/v1"
            ollama_profile = OpenAIModelProfile(
                json_schema_transformer=OpenAIJsonSchemaTransformer,
                supports_json_schema_output=True,
                supports_json_object_output=True,
                openai_chat_thinking_field="reasoning",
                openai_chat_send_back_thinking_parts="tags",
            )
            return OpenAIModel(
                settings.OLLAMA_MODEL,
                provider=OllamaProvider(base_url=ollama_base_url),
                profile=ollama_profile,
                settings={"temperature": 0.0},
            )
        else:
            raise ValueError(f"Unknown provider: {pn}")

    def _register_tools(self, agent: Agent) -> None:
        """Register agent tools for PubMed search with PICO decomposition and COT reasoning."""
        
        @agent.tool
        async def log_reasoning_step(
            ctx: RunContext[MedicalAgentDeps],
            phase: str,
            thought: str,
            details: str | None = None,
        ) -> str:
            """
            Log a step in your chain of thought reasoning process.
            
            Use this tool to document your thinking at each phase:
            - UNDERSTANDING: What is the user asking?
            - PLANNING: How will you search for information?
            - SEARCHING: What did you find?
            - SYNTHESIZING: How are you combining the evidence?
            
            Args:
                ctx: Run context with dependencies.
                phase: One of 'UNDERSTANDING', 'PLANNING', 'SEARCHING', 'SYNTHESIZING'.
                thought: Your main thought or conclusion for this phase.
                details: Optional additional details or sub-steps.
            
            Returns:
                Confirmation message.
            """
            step = {
                "phase": phase.upper(),
                "thought": thought,
                "details": details,
            }
            ctx.deps.reasoning_log.append(step)
            
            logger.info(
                f"COT {phase}",
                extra={"thought": thought[:100], "details": details[:100] if details else None}
            )
            
            return f"Logged {phase} reasoning step."
        
        @agent.tool
        async def search_pubmed_pico(
            ctx: RunContext[MedicalAgentDeps],
            population_terms: list[str],
            intervention_terms: list[str],
            modifier_terms: list[str] | None = None,
            outcome_terms: list[str] | None = None,
            recent_years: int | None = None,
        ) -> str:
            """
            Search PubMed using PICO-structured query for high-precision results.
            
            Use this tool for medical literature search. Break down the user's
            question into PICO components:
            - population_terms: Disease, condition, or patient population
            - intervention_terms: Treatment, diagnostic method, or topic
            - modifier_terms: Stage, severity, age group, etc. (optional)
            - outcome_terms: Endpoints like survival, efficacy (optional)
            - recent_years: Limit to recent N years if user wants latest research
            
            Args:
                ctx: Run context with dependencies.
                population_terms: List of disease/condition terms.
                intervention_terms: List of treatment/topic terms.
                modifier_terms: Optional modifier terms (stage II, pediatric, etc).
                outcome_terms: Optional outcome terms.
                recent_years: Optional filter to last N years.
            
            Returns:
                Formatted search results with article summaries.
            """
            print("\n" + "=" * 50)
            print("🛑 TOOL TRIGGERED: search_pubmed_pico")
            print(f"Population: {population_terms}")
            print(f"Intervention: {intervention_terms}")
            print("=" * 50 + "\n")
            logger.info(
                "search_pubmed_pico called",
                extra={
                    "population_terms": population_terms,
                    "intervention_terms": intervention_terms,
                    "modifier_terms": modifier_terms or [],
                    "outcome_terms": outcome_terms or [],
                    "recent_years": recent_years,
                },
            )

            import datetime
            
            # Build date range if recent_years specified
            date_range = None
            if recent_years and recent_years > 0:
                current_year = datetime.datetime.now().year
                date_range = (current_year - recent_years, current_year)
            
            # Create PICO query
            pico = PICOQuery(
                population=population_terms,
                intervention=intervention_terms,
                modifiers=modifier_terms or [],
                outcome=outcome_terms or [],
                date_range=date_range,
                humans_only=True,
            )
            
            # Build queries and execute fallbacks
            query_builder = PubMedQueryBuilder()
            fallback_queries = query_builder.build_fallback_queries(pico)
            logger.info(
                "Built fallback query cascade: %s",
                " || ".join(q[:140] for q in fallback_queries),
            )
            
            articles = []
            successful_query = ""
            
            # Execute cascade search logic
            for q in fallback_queries:
                logger.info(
                    "Attempting query fallback level",
                    extra={"query_preview": q[:160]},
                )
                try:
                    articles = await asyncio.to_thread(ctx.deps.pubmed_client.search, q)
                    if articles:
                        successful_query = q
                        logger.info(
                            "Query successful",
                            extra={"articles_retrieved": len(articles)},
                        )
                        break  # Found papers, stop relaxing query!
                except Exception as e:
                    logger.warning("Error querying PubMed", extra={"error": str(e)})
                    continue

            # Log the successful query for real-time display
            ctx.deps.query_log.append({
                "type": "PICO",
                "population": population_terms,
                "intervention": intervention_terms,
                "modifiers": modifier_terms or [],
                "outcomes": outcome_terms or [],
                "recent_years": recent_years,
                "built_query": successful_query if successful_query else "No papers found in fallback chain",
            })
            
            logger.info(
                "PICO search completed",
                extra={
                    "population": population_terms,
                    "intervention": intervention_terms,
                    "optimized_query": successful_query[:200] if successful_query else "Failed",
                    "articles_retrieved": len(articles)
                }
            )
            
            try:
                ctx.deps.retrieved_articles.extend(articles)
                
                if not articles:
                    # Fallback to simpler query if no results
                    simple_query = " ".join(population_terms + intervention_terms)
                    logger.info(
                        "Fallback to simple query",
                        extra={"simple_query": simple_query[:160]},
                    )
                    articles = await asyncio.to_thread(ctx.deps.pubmed_client.search, simple_query)
                    ctx.deps.retrieved_articles.extend(articles)
                    
                    if not articles:
                        return "No relevant PubMed articles found for this query."
                
                # Format results for LLM context
                query_preview = successful_query[:150] if successful_query else "No query succeeded"
                results = [f"**Query used:** `{query_preview}...`\n"]
                for article in articles[:3]:  # Limit to top 3 and keep snippets short
                    results.append(article.to_compact_context())
                
                return "\n---\n".join(results)
                
            except Exception as e:
                logger.error("PubMed search failed", extra={"error": str(e)})
                return f"PubMed search encountered an error: {str(e)}"
        
        @agent.tool
        async def search_pubmed_simple(
            ctx: RunContext[MedicalAgentDeps],
            query: str
        ) -> str:
            """
            Simple PubMed search with a free-text query.
            
            Use this only for simple, specific searches. For medical questions,
            prefer search_pubmed_pico for better results.
            
            Args:
                ctx: Run context with dependencies.
                query: Free-text search query.
            
            Returns:
                Formatted search results.
            """
            logger.info("Simple PubMed search", extra={"query": query})
            
            try:
                articles = await asyncio.to_thread(ctx.deps.pubmed_client.search, query)
                ctx.deps.retrieved_articles.extend(articles)

                logger.info(
                    "Simple PubMed search complete",
                    extra={"articles_retrieved": len(articles)},
                )
                
                if not articles:
                    return "No relevant PubMed articles found for this query."
                
                results = []
                for article in articles[:3]:
                    results.append(article.to_compact_context())
                
                return "\n---\n".join(results)
                
            except Exception as e:
                logger.error("PubMed search failed", extra={"error": str(e)})
                return f"PubMed search encountered an error: {str(e)}"
    
    async def answer_query(
        self,
        query: str,
        chat_history: Optional[List[dict]] = None,
    ) -> MedicalQueryResult:
        """
        Answer a medical query using PubMed context with a provider cascade fallback.
        """
        logger.info("Processing medical query", extra={"query": query[:100]})
        
        # Build prompt
        prompt = f"""User Query: {query}

Please search PubMed for relevant literature and provide an evidence-based response.
Include PMID citations for any claims you make based on the search results."""

        is_eval_prompt = "The final answer is: [Letter]" in query
        if is_eval_prompt:
            prompt += (
                "\n\nBenchmark mode: reply with a single option letter only. "
                "Do not include explanations or extra text. The last line must be exactly: "
                "The final answer is: [Letter]."
            )
        else:
            prompt += (
                "\n\nAnswer concisely, cite PMIDs inline, and avoid filler or conversational preambles."
            )

        deps = MedicalAgentDeps(
            pubmed_client=self.pubmed_client,
            user_query=query,
        )

        last_error = None
        provider_errors: list[str] = []
        question_deadline = time.monotonic() + 900
        for provider_name in settings.PROVIDER_CASCADE:
            logger.info(f"Attempting query with provider: {provider_name}")
            attempt = 1
            rate_limit_cooldowns = 0
            while attempt <= 5:
                if time.monotonic() > question_deadline:
                    raise GeminiAPIError(
                        f"Question exceeded the adaptive retry deadline while using provider {provider_name}."
                    )
                try:
                    current_model = self._get_model_instance(provider_name)

                    # Keep full RAG/tool-calling unless Ollama tools are explicitly disabled.
                    use_tools = not (
                        provider_name.lower() == "ollama" and not settings.OLLAMA_ENABLE_TOOLS
                    )

                    run_prompt = prompt
                    attempt_deps = MedicalAgentDeps(
                        pubmed_client=self.pubmed_client,
                        user_query=query,
                    )

                    if not use_tools:
                        # For models without tool-calling support, pre-retrieve context and inject it.
                        search_seed = query
                        if "Options:" in search_seed:
                            search_seed = search_seed.split("Options:", 1)[0]
                        if "User Query:" in search_seed:
                            search_seed = search_seed.split("User Query:", 1)[1]
                        search_seed = search_seed.strip()

                        try:
                            prefetched_articles = await asyncio.to_thread(
                                self.pubmed_client.search,
                                search_seed,
                            )
                        except Exception as e:
                            logger.warning(
                                "Pre-retrieval failed in no-tools mode: %s",
                                str(e),
                            )
                            prefetched_articles = []

                        if prefetched_articles:
                            attempt_deps.retrieved_articles.extend(prefetched_articles)
                            context_blocks = [
                                article.to_compact_context() for article in prefetched_articles[:3]
                            ]
                            retrieved_context = "\n---\n".join(context_blocks)
                        else:
                            retrieved_context = "No PubMed context retrieved."

                        run_prompt = (
                            f"{prompt}\n\n"
                            "Provider note: Tool-calling is disabled for this run. "
                            "Use the retrieved PubMed context below as evidence, cite PMIDs inline when present, "
                            "answer with a single option letter only, and end exactly with: The final answer is: [Letter].\n\n"
                            f"Retrieved PubMed Context:\n{retrieved_context}"
                        )

                    # Initialize agent
                    agent = Agent(
                        model=current_model,
                        system_prompt=MEDICAL_SYSTEM_PROMPT,
                        deps_type=MedicalAgentDeps,
                        output_type=str,
                        model_settings={"temperature": 0.0, "timeout": 180},
                    )

                    # Register tools
                    if use_tools:
                        self._register_tools(agent)

                    print(f"\n🚀 SENDING TO: {provider_name.upper()} (Attempt {attempt}/5)")

                    usage_limits = self._build_usage_limits(run_prompt, use_tools)
                    if usage_limits is not None:
                        print(f"🔧 Tool-call budget: {usage_limits.tool_calls_limit}")
                        logger.info(
                            "Tool-call budget set",
                            extra={"tool_calls_limit": usage_limits.tool_calls_limit},
                        )
                    logger.info(
                        "Provider attempt starting",
                        extra={
                            "provider": provider_name,
                            "attempt": attempt,
                            "use_tools": use_tools,
                            "query_preview": query[:120],
                        },
                    )

                    # Run the agent
                    result = await asyncio.wait_for(
                        agent.run(
                            run_prompt,
                            deps=attempt_deps,
                            usage_limits=usage_limits,
                        ),
                        timeout=240,
                    )

                    print(f"✅ SUCCESSFUL RESPONSE FROM {provider_name.upper()}")
                    print(f"✅ SUCCESSFUL RAW OUTPUT FROM {provider_name.upper()}: ")
                    print(getattr(result, "data", result.output))
                    print("-" * 40)

                    # Extract PMIDs from retrieved articles
                    sources = [article.pmid for article in attempt_deps.retrieved_articles]
                    logger.info(
                        "Provider attempt completed",
                        extra={
                            "provider": provider_name,
                            "attempt": attempt,
                            "sources_count": len(sources),
                            "queries_logged": len(attempt_deps.query_log),
                            "reasoning_steps": len(attempt_deps.reasoning_log),
                        },
                    )

                    logger.info(f"Query successfully answered by provider: {provider_name}")

                    return MedicalQueryResult(
                        answer=result.output,
                        sources=sources,
                        query_log=attempt_deps.query_log,
                        reasoning_steps=attempt_deps.reasoning_log,
                        confidence="high" if sources else "low",
                        provider_used=provider_name,
                    )

                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    is_rate_limit = (
                        "429" in error_str
                        or "quota" in error_str.lower()
                        or "RESOURCE_EXHAUSTED" in error_str
                    )

                    if is_rate_limit:
                        rate_limit_cooldowns += 1
                        wait_time = 65 + random.uniform(0, 5)
                        if time.monotonic() + wait_time > question_deadline:
                            raise GeminiAPIError(
                                f"Question would exceed the adaptive retry deadline while cooling down provider {provider_name}."
                            )
                        print(f"\n⚠️ RATE LIMIT HIT (429). Adaptive cooldown triggered.")
                        print(f"⏳ Pausing pipeline for {wait_time:.2f} seconds to let quota refill...")
                        await asyncio.sleep(wait_time)
                        print("🔄 Resuming pipeline...")
                        continue

                    print(f"\n❌ CRASH IN PROVIDER: {provider_name.upper()}")
                    print(f"Exception Type: {type(e).__name__}")
                    print(f"Exception Details: {error_str[:150]}...")
                    print("-" * 40)
                    logger.warning(
                        "%s attempt %s failed: %s",
                        provider_name,
                        attempt,
                        error_str,
                    )
                    if attempt < 5:
                        backoff_time = (2 ** attempt) * 2 + random.uniform(0, 1.5)
                        print(f"⏳ Non-quota error. Retrying in {backoff_time} seconds...")
                        await asyncio.sleep(backoff_time)
                        attempt += 1
                        continue

                    break

            logger.warning(f"{provider_name} failed, falling back to next model...")
            continue
        # If we exhausted the cascade
        logger.error("All providers in the cascade failed.")
        error_summary = " | ".join(provider_errors) if provider_errors else str(last_error)
        raise GeminiAPIError(
            f"Failed to process query after trying all providers. Details: {error_summary}"
        ) from last_error
    
    async def chat(
        self,
        message: str,
        history: Optional[List[dict]] = None,
    ) -> tuple[str, List[dict], List[dict]]:
        """
        Chat interface for Streamlit integration.
        
        Args:
            message: User message.
            history: Chat history (not used in simple implementation).
        
        Returns:
            Tuple of (response string, query_log list, reasoning_steps list).
        """
        result = await self.answer_query(message, history)
        
        # Format response with sources
        response = result.answer
        
        if result.sources:
            response += f"\n\n**Sources:** "
            response += ", ".join([f"PMID:{pmid}" for pmid in result.sources[:5]])
        
        response += f"\n\n*{result.disclaimer}*"
        
        return response, result.query_log, result.reasoning_steps
