from src.config import settings

class ModelRegistry:
    """
    Central registry for LLM instances.
    Supports: Colab (remote GPU), Ollama (local), and Gemini (API).
    """
    
    @staticmethod
    def get_llm(temperature: float = 0.0, json_mode: bool = False):
        """Standard/Legacy accessor - proxies to Smart LLM."""
        return ModelRegistry.get_smart_llm(temperature, json_mode)

    @staticmethod
    def _get_colab_llm(temperature: float = 0.0, json_mode: bool = False):
        """Get LLM from Colab server (OpenAI-compatible API)."""
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            base_url=f"{settings.COLAB_API_URL}/v1",
            api_key="not-needed",  # Colab server doesn't require auth
            model=settings.OLLAMA_SMART_MODEL,  # Bug Fix #14: Use settings instead of hardcoded
            temperature=temperature,
            max_tokens=512
        )

    @staticmethod
    def get_smart_llm(temperature: float = 0.0, json_mode: bool = False):
        """
        Get 'Smart' LLM for complex tasks (Planner, Extractor, QA).
        
        Priority: Colab > Ollama > Gemini
        """
        # Option 1: Use Colab (remote GPU)
        if settings.USE_COLAB and settings.COLAB_API_URL:
            return ModelRegistry._get_colab_llm(temperature, json_mode)
        
        # Option 2: Use Ollama (local)
        if settings.USE_OLLAMA:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                from langchain_community.chat_models import ChatOllama
                
            return ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_SMART_MODEL,
                temperature=temperature,
                keep_alive="5m",
                format="json" if json_mode else None
            )
        
        # Option 3: Use Gemini (API)
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,
            max_retries=settings.RETRY_MAX_ATTEMPTS
        )

    @staticmethod
    def get_fast_llm(temperature: float = 0.0, json_mode: bool = False):
        """
        Get 'Fast' LLM for simple tasks (Step Definer).
        
        Uses local Ollama even when Colab is enabled (to reduce latency).
        """
        if settings.USE_OLLAMA:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                from langchain_community.chat_models import ChatOllama
            
            return ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_FAST_MODEL,
                temperature=temperature,
                keep_alive="5m",
                format="json" if json_mode else None
            )
        else:
            # Fallback to Smart if Ollama not available
            return ModelRegistry.get_smart_llm(temperature, json_mode)

    @staticmethod
    def get_light_llm(temperature: float = 0.0, json_mode: bool = False):
        """Alias for get_fast_llm."""
        return ModelRegistry.get_fast_llm(temperature, json_mode)

    @staticmethod
    def get_heavy_llm(temperature: float = 0.0, json_mode: bool = False):
        """Alias for get_smart_llm."""
        return ModelRegistry.get_smart_llm(temperature, json_mode)

    @staticmethod
    def get_flash_llm(temperature: float = 0.0, json_mode: bool = False):
        """Alias for get_fast_llm."""
        return ModelRegistry.get_fast_llm(temperature, json_mode)

