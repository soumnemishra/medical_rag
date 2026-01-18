# FILE: app.py
"""
Streamlit Chat Application for Medical RAG Chatbot.

A chat-based interface for medical doctors to query medical literature
using real-time PubMed search powered by Pydantic AI and Gemini.

Run with:
    streamlit run app.py

Environment Variables:
    GOOGLE_CLOUD_PROJECT: GCP project ID
    GEMINI_MODEL: Gemini model name (default: gemini-2.0-flash)
    PUBMED_API_KEY: Optional NCBI API key
"""

import asyncio
import logging
import re
from typing import List, Dict

import streamlit as st

# Fix for nested event loops in Streamlit
import nest_asyncio
nest_asyncio.apply()

from src.config import configure_logging, settings
from src.agent import MedicalAgent # this imported the medical agent 
from src.exceptions import TransientError, PermanentError

# Configure logging
configure_logging() # configures the loggings 
logger = logging.getLogger(__name__)


def run_async(coro):
    """Run async coroutine in Streamlit-compatible way."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def make_pmids_clickable(text: str) -> str:
    """
    Convert PMID references to clickable PubMed links.
    
    Handles formats like:
    - PMID:12345678
    - PMID: 12345678
    - (PMID:12345678)
    - PMID 12345678
    
    Args:
        text: Response text containing PMID references.
    
    Returns:
        Text with PMIDs converted to markdown links.
    """
    # Pattern matches various PMID formats
    # Captures the PMID number while preserving surrounding text
    pattern = r'PMID[:\s]*(\d{7,8})'
    
    def replace_pmid(match):
        pmid = match.group(1)
        return f'[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)'
    
    return re.sub(pattern, replace_pmid, text, flags=re.IGNORECASE)

# Page configuration
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for medical theme with thinking animation
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-left: 20%;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border: 1px solid #4a5568;
        margin-right: 20%;
    }
    
    .source-badge {
        background: #00d4ff;
        color: #1a1a2e;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .disclaimer {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Thinking animation styles */
    .thinking-container {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #6366f1;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .thinking-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #6366f1, transparent);
        animation: thinking-glow 2s infinite;
    }
    
    @keyframes thinking-glow {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .thinking-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #a5b4fc;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .thinking-phase {
        background: rgba(99, 102, 241, 0.1);
        border-left: 3px solid #6366f1;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .phase-title {
        color: #818cf8;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    
    .phase-thought {
        color: #e2e8f0;
        font-size: 0.85rem;
    }
    
    .phase-details {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.25rem;
        font-style: italic;
    }
    
    .stTextInput > div > div > input {
        background: #2d3748;
        border: 1px solid #4a5568;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #7b2cbf, #00d4ff);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = MedicalAgent()
            logger.info("Medical agent initialized in session")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            st.session_state.agent = None
            st.session_state.init_error = str(e)


def render_sidebar() -> None:
    """Render the sidebar with settings and info."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model info
        st.markdown(f"**Model:** `{settings.GEMINI_MODEL}`")
        st.markdown(f"**Max Results:** `{settings.MAX_SEARCH_RESULTS}`")
        
        st.markdown("---")
        
        # Search settings
        st.markdown("## 🔍 Search Options")
        search_years = st.slider(
            "Publication Years",
            min_value=1,
            max_value=10,
            value=5,
            help="Limit search to recent publications"
        )
        st.session_state.search_years = search_years
        
        st.markdown("---")
        
        # Info section
        st.markdown("## ℹ️ About")
        st.markdown("""
        This assistant helps medical professionals find 
        evidence-based information from PubMed literature.
        
        **Features:**
        - 🔬 Real-time PubMed search
        - 🤖 Gemini-powered responses
        - 📚 Citation tracking
        - 💬 Conversational interface
        """)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat_message(role: str, content: str) -> None:
    """Render a single chat message."""
    with st.chat_message(role):
        st.markdown(content)


def render_chat_history() -> None:
    """Render the chat history."""
    for message in st.session_state.messages:
        render_chat_message(message["role"], message["content"])


def render_thinking_process(reasoning_steps: list) -> None:
    """
    Render the Chain of Thought reasoning process in a styled expander.
    
    Args:
        reasoning_steps: List of reasoning step dictionaries with phase, thought, details.
    """
    if not reasoning_steps:
        return
    
    # Phase icons mapping
    phase_icons = {
        "UNDERSTANDING": "🧠",
        "PLANNING": "📝",
        "SEARCHING": "🔍",
        "SYNTHESIZING": "⚖️",
    }
    
    with st.expander("💡 **Thinking Process** (click to expand)", expanded=False):
        st.markdown("""
        <div class="thinking-container">
            <div class="thinking-header">
                <span>✨</span>
                <span>Chain of Thought Reasoning</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for step in reasoning_steps:
            phase = step.get("phase", "THINKING")
            thought = step.get("thought", "")
            details = step.get("details", "")
            icon = phase_icons.get(phase, "💭")
            
            st.markdown(f"""
            <div class="thinking-phase">
                <div class="phase-title">{icon} {phase}</div>
                <div class="phase-thought">{thought}</div>
                {f'<div class="phase-details">{details}</div>' if details else ''}
            </div>
            """, unsafe_allow_html=True)


async def process_query(query: str) -> tuple[str, list, list]:
    """
    Process a user query through the medical agent.
    
    Args:
        query: User's medical question.
    
    Returns:
        Tuple of (response string, query_log list, reasoning_steps list).
    """
    if st.session_state.agent is None:
        return "❌ Agent not initialized. Please check your configuration.", [], []
    
    try:
        response, query_log, reasoning_steps = await st.session_state.agent.chat(
            query,
            history=st.session_state.messages,
        )
        return response, query_log, reasoning_steps
        
    except TransientError as e:
        logger.warning(f"Transient error: {e}")
        return f"⚠️ Temporary error occurred. Please try again.\n\nDetails: {e.message}", [], []
        
    except PermanentError as e:
        logger.error(f"Permanent error: {e}")
        return f"❌ An error occurred: {e.message}", [], []
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return f"❌ Unexpected error: {str(e)}", [], []


def main() -> None:
    """Main Streamlit application entry point."""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.markdown('<h1 class="main-header">🏥 Medical RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #a0aec0;">Evidence-based medical information powered by PubMed & Gemini</p>',
        unsafe_allow_html=True
    )
    
    # Check for initialization errors
    if hasattr(st.session_state, "init_error"):
        st.error(f"⚠️ Initialization Error: {st.session_state.init_error}")
        st.info("Please check your environment variables and try again.")
        return
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> This tool provides information for educational purposes only. 
        Always verify information and use clinical judgment. Not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Render chat history
    render_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_chat_message("user", prompt)
        
        # Process query and get response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching PubMed and generating response..."):
                response, query_log, reasoning_steps = run_async(process_query(prompt))
                
                # Display Chain of Thought reasoning process
                if reasoning_steps:
                    render_thinking_process(reasoning_steps)
                
                # Display the built query in an expander
                if query_log:
                    with st.expander("🔬 **Query Builder Details** (click to expand)", expanded=False):
                        for i, q in enumerate(query_log):
                            st.markdown(f"### Search #{i+1}: {q.get('type', 'Unknown')} Query")
                            
                            # Show PICO decomposition
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Population:**")
                                st.code(", ".join(q.get("population", [])))
                                st.markdown("**Intervention:**")
                                st.code(", ".join(q.get("intervention", [])))
                            with col2:
                                st.markdown("**Modifiers:**")
                                st.code(", ".join(q.get("modifiers", [])) or "None")
                                st.markdown("**Outcomes:**")
                                st.code(", ".join(q.get("outcomes", [])) or "None")
                            
                            if q.get("recent_years"):
                                st.markdown(f"**Date Filter:** Last {q['recent_years']} years")
                            
                            # Show the built query
                            st.markdown("**Built PubMed Query:**")
                            st.code(q.get("built_query", "N/A"), language="text")
                            st.markdown("---")
                
                # Convert PMIDs to clickable links
                response = make_pmids_clickable(response)
                
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
