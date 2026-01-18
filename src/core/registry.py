from src.config import settings

class ModelRegistry:
    """
    Central registry for LLM instances.
    Supports switching between Ollama and Gemini based on configuration.
    """
    
    @staticmethod
    def get_llm(temperature: float = 0.0):
        """
        Get the configured LLM instance.
        """
        if settings.USE_OLLAMA:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=temperature,
                keep_alive="5m"
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=temperature,
                convert_system_message_to_human=True
            )
