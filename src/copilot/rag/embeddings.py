"""Embedding provider wrapper (Voyage by default; swap the body to change providers)."""
from __future__ import annotations

import voyageai

from copilot.core.settings import settings

_client = voyageai.Client(api_key=settings.voyage_api_key)


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    if not texts:
        return []
    resp = _client.embed(texts, model=settings.embed_model, input_type=input_type)
    return resp.embeddings


def embed_query(text: str) -> list[float]:
    return embed_texts([text], input_type="query")[0]


def vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
