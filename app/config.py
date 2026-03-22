"""
NLP/Text Agent — Application configuration via pydantic-settings.
Reads from .env.free (default) or .env.paid depending on deployment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the NLP/Text Agent microservice."""

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env.free"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ───────────────────────────────────────────────
    PORT: int = 8005
    APP_NAME: str = "NLP/Text Agent"
    APP_VERSION: str = "1.0.0"

    # ── LLM Provider ────────────────────────────────────────
    LLM_PROVIDER: str = "ollama"  # ollama | groq | openai

    # Ollama
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-70b-versatile"

    # OpenAI (optional)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ── NLP Models ──────────────────────────────────────────
    SPACY_MODEL: str = "en_core_web_sm"
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Processing ──────────────────────────────────────────
    NLP_BATCH_SIZE: int = 32
    MAX_TEXT_LENGTH: int = 10_000

    # ── Storage ─────────────────────────────────────────────
    HF_CACHE_DIR: str = "/app/hf_cache"
    REDIS_URL: str = "redis://redis:6379"
    VECTOR_STORE_TYPE: str = "none"  # none | pgvector
    PGVECTOR_URL: str = ""

    # ── Docker / EC2 ────────────────────────────────────────
    USE_DOCKER: bool = True
    USE_EC2: bool = False


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
