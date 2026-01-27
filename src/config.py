# FILE: src/config.py
"""
Configuration management for the Medical RAG Chatbot.

Loads environment variables and provides centralized configuration access.
All secrets and API keys are read from environment variables.

Example Usage:
    from src.config import settings
    
    print(settings.GEMINI_MODEL)
    print(settings.PUBMED_BASE_URL)
"""

import os
import logging
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Attributes:
        GOOGLE_CLOUD_PROJECT: GCP project ID for Vertex AI.
        GOOGLE_CLOUD_LOCATION: GCP region (default: us-central1).
        GEMINI_MODEL: Gemini model name to use.
        PUBMED_BASE_URL: PubMed E-utilities base URL.
        PUBMED_API_KEY: Optional NCBI API key for higher rate limits.
        MAX_SEARCH_RESULTS: Maximum number of PubMed results to fetch.
        RETRY_MAX_ATTEMPTS: Max retry attempts for transient errors.
        RETRY_WAIT_SECONDS: Initial wait time for exponential backoff.
    """
    
    # Google API Configuration
    GOOGLE_API_KEY: Optional[str] = Field(
        default=None,
        description="Google AI Studio API key"
    )
    
    # GCP Configuration (for Vertex AI)
    GOOGLE_CLOUD_PROJECT: str = Field(
        default="your-gcp-project-id",
        description="GCP project ID"
    )
    GOOGLE_CLOUD_LOCATION: str = Field(
        default="us-central1",
        description="GCP region for Vertex AI"
    )
    
    # Gemini Configuration
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model to use"
    )

    # Ollama Configuration
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    OLLAMA_MODEL: str = Field(
        default="phi4-mini:latest",
        description="Default Ollama model"
    )
    OLLAMA_SMART_MODEL: str = Field(
        default="llama3.2:3b",
        description="Smart Ollama model for Planner/QA/Extractor"
    )
    OLLAMA_FAST_MODEL: str = Field(
        default="qwen2.5:1.5b",
        description="Fast Ollama model for Step Definer"
    )
    USE_OLLAMA: bool = Field(
        default=True,
        description="Use Ollama instead of Gemini"
    )
    
    # Colab Configuration (for hybrid architecture)
    USE_COLAB: bool = Field(
        default=False,
        description="Use Google Colab for heavy compute tasks"
    )
    COLAB_API_URL: str = Field(
        default="",
        description="ngrok URL from Colab server (e.g., https://xxxx.ngrok-free.app)"
    )
    USE_HYBRID: bool = Field(
        default=False,
        description="Use Hybrid (Gemini for Heavy, Ollama for Light) agents"
    )
    
    # Retrieval Configuration
    USE_MEDCPT: bool = Field(
        default=False,
        description="Use MedCPT for biomedical retrieval (requires GPU, ~4GB VRAM)"
    )
    USE_HYBRID_RETRIEVAL: bool = Field(
        default=True,
        description="Use hybrid retrieval (BM25 + Dense with RRF)"
    )
    
    # PubMed Configuration
    PUBMED_BASE_URL: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        description="PubMed E-utilities base URL"
    )
    PUBMED_API_KEY: Optional[str] = Field(
        default=None,
        description="NCBI API key for higher rate limits"
    )
    MAX_SEARCH_RESULTS: int = Field(
        default=10,
        description="Maximum PubMed results per query"
    )
    
    # Retry Configuration
    RETRY_MAX_ATTEMPTS: int = Field(
        default=10,
        description="Maximum retry attempts"
    )
    RETRY_WAIT_SECONDS: float = Field(
        default=4.0,
        description="Initial wait time for backoff"
    )
    
    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """
    Get application settings singleton.
    
    Returns:
        Settings: Validated settings instance.
    """
    # Bug Fix #9: Return singleton instead of creating new instance
    return settings


# Global settings instance (singleton)
settings = Settings()


def configure_logging() -> None:
    """Configure structured logging for the application."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.info("Logging configured", extra={"level": settings.LOG_LEVEL})
