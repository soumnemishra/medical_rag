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
from dataclasses import dataclass, field #data class 
from typing import List, Optional


# basemodel verifies the structure of the data it is a pydantic 
#base model is the data validation system 
from pydantic import BaseModel, Field #base model is only used for the output classes not for shared 


#  agent wraps the llm and run context gives tools to access the shared memory 
from pydantic_ai import Agent, RunContext 

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
MEDICAL_SYSTEM_PROMPT = """You are a helpful medical information assistant for healthcare professionals.

Your role is to:
1. Provide accurate, evidence-based medical information
2. Search PubMed using PICO-structured queries for best results
3. Cite PubMed sources when available
4. Be clear about limitations and uncertainties

## CHAIN OF THOUGHT REASONING
Before answering any medical query, you MUST think step-by-step using the log_reasoning_step tool.
Always log your reasoning in these phases:

1. **UNDERSTANDING** - Log what you understand about the user's question:
   - What is the clinical question?
   - What type of information is being requested (treatment, diagnosis, prognosis, etc.)?
   - Are there any constraints (patient population, timeline, etc.)?

2. **PLANNING** - Log your search strategy:
   - How will you decompose this into PICO components?
   - What synonyms or related terms should you include?
   - Should you filter by recency?

3. **SEARCHING** - Execute the search and log what you found:
   - How many relevant articles were found?
   - What are the key themes in the results?

4. **SYNTHESIZING** - Log how you're combining the evidence:
   - What are the main findings across studies?
   - Are there any conflicting results?
   - What is the overall confidence level?

SEARCH STRATEGY - Always use search_pubmed_pico for medical queries:
1. Extract POPULATION terms: disease, condition, patient group (e.g., "melanoma", "type 2 diabetes")
2. Extract INTERVENTION terms: treatment, diagnostic, topic (e.g., "immunotherapy", "diagnosis")
3. Extract MODIFIER terms: stage, severity, age (e.g., "stage II", "pediatric")
4. Extract OUTCOME terms if present: survival, efficacy, response rate
5. Set recent_years if user asks for "latest" or "recent" (typically 5 years)

Example decomposition for "What is the latest treatment for stage II melanoma?":
- population_terms: ["melanoma", "cutaneous melanoma"]
- intervention_terms: ["treatment", "therapy"]
- modifier_terms: ["stage II", "stage 2"]
- recent_years: 5

Guidelines:
- Always base answers on retrieved PubMed articles
- **CITATION FORMAT**: You MUST cite PMIDs inline with each specific claim, not just at the end
  - Place citations immediately after each factual statement: "Treatment X showed 40% improvement (PMID:12345678)."
  - When multiple sources support a claim, cite all: "...is recommended (PMID:12345678, PMID:87654321)."
  - Do NOT lump all citations at the end - distribute them throughout the response
- Use medical terminology appropriate for healthcare professionals
- Be concise but thorough
- Acknowledge when evidence is limited or conflicting
- Never provide specific treatment recommendations for individual patients

IMPORTANT: You are providing information to medical professionals, not patients.
Always include relevant INLINE citations and maintain scientific accuracy."""




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
        import os #import os to set environment variables AND INGORE GLOBAL PROBLEMS 
        '''ALLOWS FLEXIBLE INJECTION OF A PUB MED HANDLER ENABLING TESTABILITY AND SAFE RESULTS 
        
        JUST LIKE CREATING A ENGINE OR A DEFAULT ENGINE '''
        self.pubmed_client = pubmed_client or PubMedClient() #SET UP THE PUBMED CLIENT
        
        
        # Set up Vertex AI configuration

        project = settings.GOOGLE_CLOUD_PROJECT
        location = settings.GOOGLE_CLOUD_LOCATION
        
        if project == "your-gcp-project-id":
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT not configured. Please set it in your .env file.\n"
                "Example: GOOGLE_CLOUD_PROJECT=my-project-123"
            )
        
        # Ensure environment variables are set for Vertex AI
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project) #ENSURE THE ENVIRONMENT VARIABLES ARE SET
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)
        
        # Build Vertex AI model string
        # Format: google-vertex:model-name
        #IF BREAKS WE CANNOT GET THE INFO ABOUT THE MODEL BEING USED
        model_string = f"google-vertex:{settings.GEMINI_MODEL}" # this tells which model being used 
        
        logger.info(
            "Initializing Vertex AI agent",
            extra={
                "project": project,
                "location": location,
                "model": settings.GEMINI_MODEL,
            }
        )
        
        # Initialize Pydantic AI agent with Vertex AI
        #MAIN AGENT CODE WHERE THE AGENT IS BEIN BUILD UP 
        '''GET THE INFO OF THE MODEL AND THE SYSTEM PROMPT AND THE DEPENDENCIES AND THE OUTPUT TYPE
        DEPS_TYPE TELLS THE LLM FOR EVERY RUN, CREATE ONE SHARED MEMORY OBJECT OF THIS TYPE AND PASS IT
        TO ALL THE TOOLS'''
        self.agent = Agent(
            model=model_string, #WHICH MODEL 
            system_prompt=MEDICAL_SYSTEM_PROMPT, #SYSTEM PROMPT HOW THE MODEL SHOULD BEHAVE 
            # TELLS THE LLM For every run, create ONE shared memory object of this type
            deps_type=MedicalAgentDeps,
            output_type=str,  # LLM GENERATE THE RAW TEXT THEN WE PREPROCESS IT TO MEDQUERY RESULT 
        )
        
        # Query builder for PICO-based searches
        #converts the pico query to pubmed query 
        self.query_builder = PubMedQueryBuilder() #
        
        # Register tools
        '''Tools belong to agent behavior

            Registered once
            Not recreated per query'''
        self._register_tools() #REGISTER TOOL TO THE AGENT  WHAT LLM CAN ACT 
        # this log which model is been running and what is the model name 
        logger.info(
            "Medical agent initialized",
            extra={"model": settings.GEMINI_MODEL}
        )
    # ########################################### tOOLS 
    # this genrally create a function of register tools and returns nothing associated with the medical
    # agent ######################################################
    ##########################TOOL_1#################
    def _register_tools(self) -> None:
        """Register agent tools for PubMed search with PICO decomposition and COT reasoning."""
        
        @self.agent.tool
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
        
        @self.agent.tool
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
            
            # Build optimized query
            query_builder = PubMedQueryBuilder()
            optimized_query = query_builder.build_query(pico)
            
            # Log the query for real-time display
            ctx.deps.query_log.append({
                "type": "PICO",
                "population": population_terms,
                "intervention": intervention_terms,
                "modifiers": modifier_terms or [],
                "outcomes": outcome_terms or [],
                "recent_years": recent_years,
                "built_query": optimized_query,
            })
            
            logger.info(
                "PICO search",
                extra={
                    "population": population_terms,
                    "intervention": intervention_terms,
                    "optimized_query": optimized_query[:200]
                }
            )
            
            try:
                articles = ctx.deps.pubmed_client.search(optimized_query)
                ctx.deps.retrieved_articles.extend(articles)
                
                if not articles:
                    # Fallback to simpler query if no results
                    simple_query = " ".join(population_terms + intervention_terms)
                    articles = ctx.deps.pubmed_client.search(simple_query)
                    ctx.deps.retrieved_articles.extend(articles)
                    
                    if not articles:
                        return "No relevant PubMed articles found for this query."
                
                # Format results for LLM context
                results = [f"**Query used:** `{optimized_query[:150]}...`\n"]
                for article in articles[:5]:  # Limit to top 5
                    results.append(article.to_context_string())
                
                return "\n---\n".join(results)
                
            except Exception as e:
                logger.error("PubMed search failed", extra={"error": str(e)})
                return f"PubMed search encountered an error: {str(e)}"
        
        @self.agent.tool
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
                articles = ctx.deps.pubmed_client.search(query)
                ctx.deps.retrieved_articles.extend(articles)
                
                if not articles:
                    return "No relevant PubMed articles found for this query."
                
                results = []
                for article in articles[:5]:
                    results.append(article.to_context_string())
                
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
        Answer a medical query using PubMed context.
        
        Args:
            query: The medical question from the user.
            chat_history: Optional list of previous messages.
        
        Returns:
            MedicalQueryResult with answer and sources.
        
        Raises:
            GeminiAPIError: On LLM API failure.
        """
        logger.info("Processing medical query", extra={"query": query[:100]})
        
        # Create dependencies
        deps = MedicalAgentDeps(
            pubmed_client=self.pubmed_client,
            user_query=query,
        )
        
        try:
            # Build prompt with optional search instruction
            prompt = f"""User Query: {query}

Please search PubMed for relevant literature and provide an evidence-based response.
Include PMID citations for any claims you make based on the search results."""

            # Run the agent
            result = await self.agent.run(prompt, deps=deps)
            
            # Extract PMIDs from retrieved articles
            sources = [article.pmid for article in deps.retrieved_articles]
            
            return MedicalQueryResult(
                answer=result.output,
                sources=sources,
                query_log=deps.query_log,
                reasoning_steps=deps.reasoning_log,
                confidence="high" if sources else "low",
            )
            
        except Exception as e:
            logger.error("Agent execution failed", extra={"error": str(e)})
            raise GeminiAPIError(f"Failed to process query: {e}") from e
    
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
