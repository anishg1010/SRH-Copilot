"""Environment-driven configuration shared across all agents."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-4-5"

    # Embeddings
    voyage_api_key: str = ""
    embed_model: str = "voyage-3"
    embed_dim: int = 1024

    # Database (RAG store + LangGraph checkpointer, shared)
    database_url: str = "postgresql://copilot:copilot@localhost:5432/copilot"

    # Web search
    tavily_api_key: str = ""

    # RAG tuning
    chunk_tokens: int = 500
    chunk_overlap: int = 60
    retrieve_k: int = 6


settings = Settings()
