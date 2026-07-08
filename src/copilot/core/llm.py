"""LLM factory — single place to construct the chat model."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from copilot.core.settings import settings


def make_llm(model: str | None = None, temperature: float = 0.3, max_tokens: int = 4096):
    return ChatAnthropic(
        model=model or settings.default_model,
        api_key=settings.anthropic_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
