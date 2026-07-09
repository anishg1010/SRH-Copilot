"""Central configuration for the University AI Copilot.

Single place to tune everything. Values default here; override via .env.
Import the singleton:  from copilot.core.config import config
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ─────────────────────────────────────────────────────────────
    # EMBEDDINGS
    # ─────────────────────────────────────────────────────────────
    # Provider: "local" (sentence-transformers, DEFAULT), "chroma" (ONNX fallback), "voyage".
    embed_provider: str = "local"
    # Best multilingual (DE+EN) model; 1024-dim. GPU-accelerated on CUDA.
    local_embed_model: str = "intfloat/multilingual-e5-large"
    embed_device: str = ""            # "" = auto (cuda if available), or "cuda"/"cpu"
    embed_batch_size: int = 64        # raise on a big GPU for faster ingest
    voyage_api_key: str = ""
    voyage_embed_model: str = "voyage-3"

    # ─────────────────────────────────────────────────────────────
    # VECTOR STORE (ChromaDB — local, persistent)
    # ─────────────────────────────────────────────────────────────
    chroma_dir: str = "./chroma_db"
    chroma_collection_prefix: str = "copilot"

    # ─────────────────────────────────────────────────────────────
    # CHUNKING (re-ingest after changing)
    # ─────────────────────────────────────────────────────────────
    chunk_tokens: int = 500
    chunk_overlap: int = 60

    # ─────────────────────────────────────────────────────────────
    # DATA
    # ─────────────────────────────────────────────────────────────
    # Prefer the function-categorized corpus; fall back to the raw teaching folder.
    teaching_data_dir: str = "./data_categorized"
    teaching_data_fallback: str = "./data/teaching"

    # ─────────────────────────────────────────────────────────────
    # RETRIEVAL
    # ─────────────────────────────────────────────────────────────
    retrieve_k: int = 6
    min_score: float = 0.0

    # ─────────────────────────────────────────────────────────────
    # LLM  (open-source, run in-process via transformers — free, GPU)
    # ─────────────────────────────────────────────────────────────
    # Provider: "transformers" (DEFAULT, in-venv), "ollama", or "anthropic" (later).
    llm_provider: str = "transformers"
    hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_device: str = ""              # "" = auto (cuda), or "cuda"/"cpu"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.3

    # Ollama (only if llm_provider == "ollama")
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"

    # Anthropic (only if llm_provider == "anthropic", later)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    # ─────────────────────────────────────────────────────────────
    # WEB SEARCH (agentic layer only)
    # ─────────────────────────────────────────────────────────────
    tavily_api_key: str = ""


config = Config()
