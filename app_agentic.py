# FILE: app_agentic.py
"""
Streamlit Chat Application for Agentic Medical RAG Chatbot.

Uses the new multi-agent LangGraph-based pipeline with:
- Planner Agent: PICO decomposition
- Retriever Agent: PubMed search
- Extractor Agent: Noise filtering
- Synthesizer Agent: Answer generation with confidence

Run with:
    streamlit run app_agentic.py
"""

import asyncio
import logging
import re
from typing import List, Dict, Any

import streamlit as st

# Fix for nested event loops in Streamlit
import nest_asyncio
nest_asyncio.apply()

from src.config import configure_logging, settings
from src.exceptions import TransientError, PermanentError

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def run_async(coro):
    """Run async coroutine in Streamlit-compatible way."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def make_pmids_clickable(text: str) -> str:
    """Convert PMID references to clickable PubMed links."""
    pattern = r'PMID[:\s]*(\d{7,8})'
    
    def replace_pmid(match):
        pmid = match.group(1)
        return f'[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)'
    
    return re.sub(pattern, replace_pmid, text, flags=re.IGNORECASE)


# Page configuration
st.set_page_config(
    page_title="Agentic Medical RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for agentic theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #00ff88, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    
    .agent-status {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
    }
    
    .agent-phase {
        background: rgba(99, 102, 241, 0.1);
        border-left: 3px solid #6366f1;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .confidence-high { color: #00ff88; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #ff4444; }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agentic_graph" not in st.session_state:
        try:
            from src.orchestrator.graph import build_agentic_graph
            from src.agents.planner import planner_node_sync
            from src.agents.retriever import retriever_node_sync
            from src.agents.extractor import extractor_node_sync
            from src.agents.synthesizer import synthesizer_node_sync
            
            # Build graph with real agents
            st.session_state.agentic_graph = build_agentic_graph(
                planner=planner_node_sync,
                retriever=retriever_node_sync,
                extractor=extractor_node_sync,
                synthesizer=synthesizer_node_sync,
            )
            logger.info("Agentic RAG graph initialized")
        except Exception as e:
            logger.error(f"Failed to initialize agentic graph: {e}")
            st.session_state.agentic_graph = None
            st.session_state.init_error = str(e)


def render_sidebar() -> None:
    """Render the sidebar with settings and info."""
    with st.sidebar:
        st.markdown("## 🧠 Agentic RAG")
        
        st.markdown("""
        **Pipeline:**
        1. 🎯 Planner (PICO)
        2. 🔍 Retriever (PubMed)
        3. 📝 Extractor (Filter)
        4. 💡 Synthesizer (Answer)
        """)
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 🤖 Models")
        st.markdown("**Planner:** `phi4-mini` (Ollama)")
        st.markdown("**Extractor:** `phi4-mini` (Ollama)")
        st.markdown("**Synthesizer:** `llama3.2:3b` (Ollama)")
        st.markdown("**Fallback:** `gemini-2.0-flash`")
        
        st.markdown("---")
        
        # Settings
        max_iterations = st.slider(
            "Max Backtrack Iterations",
            min_value=1,
            max_value=5,
            value=2,
            help="Maximum retry attempts when confidence is low"
        )
        st.session_state.max_iterations = max_iterations
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_reasoning_trace(trace: List[Dict]) -> None:
    """Render the agentic reasoning trace."""
    if not trace:
        return
    
    phase_icons = {
        "PLAN": "🎯",
        "RETRIEVE": "🔍",
        "EXTRACT": "📝",
        "SYNTHESIZE": "💡",
        "BACKTRACK": "🔄",
    }
    
    with st.expander("🧠 **Agentic Reasoning Trace** (click to expand)", expanded=False):
        for step in trace:
            phase = step.get("phase", "UNKNOWN")
            thought = step.get("thought", "")
            details = step.get("details", "")
            iteration = step.get("iteration", 0)
            icon = phase_icons.get(phase, "💭")
            
            st.markdown(f"""
            <div class="agent-phase">
                <strong>{icon} {phase}</strong> (iter {iteration})<br>
                <span style="color: #e2e8f0;">{thought}</span>
                {f'<br><span style="color: #94a3b8; font-size: 0.85rem;">{details[:150]}...</span>' if details else ''}
            </div>
            """, unsafe_allow_html=True)


def render_retrieved_docs(docs: List[Dict]) -> None:
    """Render retrieved documents."""
    if not docs:
        return
    
    with st.expander(f"📚 **Retrieved Documents ({len(docs)})** (click to expand)", expanded=False):
        for doc in docs[:5]:
            pmid = doc.get("pmid", "N/A")
            title = doc.get("title", "No title")
            year = doc.get("year", "")
            
            st.markdown(f"""
            **[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)** ({year})
            
            {title[:150]}...
            
            ---
            """)


async def process_agentic_query(query: str) -> Dict[str, Any]:
    """Process a query through the agentic RAG pipeline."""
    from src.orchestrator.state import create_initial_state
    
    if st.session_state.agentic_graph is None:
        return {
            "final_answer": "❌ Agentic graph not initialized.",
            "final_confidence": 0,
            "reasoning_trace": [],
            "retrieved_docs": [],
        }
    
    try:
        # Create initial state
        initial_state = create_initial_state(query)
        initial_state["max_iterations"] = st.session_state.get("max_iterations", 2)
        
        # Run graph
        result = st.session_state.agentic_graph.invoke(initial_state)
        
        return result
        
    except Exception as e:
        logger.exception(f"Agentic query failed: {e}")
        return {
            "final_answer": f"❌ Error: {str(e)}",
            "final_confidence": 0,
            "reasoning_trace": [{"phase": "ERROR", "thought": str(e)}],
            "retrieved_docs": [],
        }


def get_confidence_class(confidence: int) -> str:
    """Get CSS class for confidence level."""
    if confidence >= 7:
        return "confidence-high"
    elif confidence >= 5:
        return "confidence-medium"
    return "confidence-low"


def main() -> None:
    """Main Streamlit application entry point."""
    init_session_state()
    render_sidebar()
    
    # Main header
    st.markdown('<h1 class="main-header">🧠 Agentic Medical RAG</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #a0aec0;">Multi-Agent Pipeline with LangGraph • Ollama + Gemini</p>',
        unsafe_allow_html=True
    )
    
    # Check for init errors
    if hasattr(st.session_state, "init_error"):
        st.error(f"⚠️ Initialization Error: {st.session_state.init_error}")
        return
    
    # Disclaimer
    st.warning("⚠️ **Disclaimer:** For educational purposes only. Always verify with clinical judgment.")
    
    st.markdown("---")
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with agentic pipeline
        with st.chat_message("assistant"):
            with st.spinner("🧠 Agentic pipeline running..."):
                result = run_async(process_agentic_query(prompt))
                
                # Show reasoning trace
                trace = result.get("reasoning_trace", [])
                render_reasoning_trace(trace)
                
                # Show retrieved docs
                docs = result.get("retrieved_docs", [])
                render_retrieved_docs(docs)
                
                # Show confidence
                confidence = result.get("final_confidence", 0)
                iterations = result.get("iteration_count", 0)
                conf_class = get_confidence_class(confidence)
                
                st.markdown(f"""
                <div class="agent-status">
                    <strong>Confidence:</strong> <span class="{conf_class}">{confidence}/10</span> | 
                    <strong>Iterations:</strong> {iterations}
                </div>
                """, unsafe_allow_html=True)
                
                # Show answer
                answer = result.get("final_answer", "No answer generated.")
                answer = make_pmids_clickable(answer)
                st.markdown(answer)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
