"""Embedding provider — multilingual-e5 (default), plus chroma/voyage options.

Default: intfloat/multilingual-e5-large via sentence-transformers — strong on the
German+English corpus, 1024-dim, GPU-accelerated. e5 REQUIRES input prefixes:
  - documents/passages → "passage: <text>"
  - queries            → "query: <text>"
We add these automatically based on input_type, which is essential for good retrieval.

Providers (EMBED_PROVIDER in .env):
  "local"  → sentence-transformers model in LOCAL_EMBED_MODEL (default e5-large)
  "chroma" → ChromaDB built-in ONNX MiniLM (English-ish, no torch) — fallback
  "voyage" → Voyage API (needs key)
"""
from __future__ import annotations

from functools import lru_cache

from copilot.core.config import config as settings


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer
    # device auto-selects CUDA if available.
    return SentenceTransformer(settings.local_embed_model, device=settings.embed_device or None)


@lru_cache(maxsize=1)
def _chroma_embedder():
    from chromadb.utils import embedding_functions
    return embedding_functions.ONNXMiniLM_L6_V2()


@lru_cache(maxsize=1)
def _voyage_client():
    import voyageai
    if not settings.voyage_api_key:
        raise RuntimeError("EMBED_PROVIDER=voyage but VOYAGE_API_KEY is not set.")
    return voyageai.Client(api_key=settings.voyage_api_key)


def _to_lists(vectors) -> list[list[float]]:
    out = []
    for v in vectors:
        if hasattr(v, "tolist"):
            v = v.tolist()
        out.append([float(x) for x in v])
    return out


def _e5_prefix(texts: list[str], input_type: str) -> list[str]:
    """Add e5's required prefixes. Only applied to e5 models."""
    if "e5" not in settings.local_embed_model.lower():
        return texts
    tag = "query: " if input_type == "query" else "passage: "
    return [tag + t for t in texts]


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed documents or queries. input_type: 'document' | 'query'."""
    if not texts:
        return []

    provider = settings.embed_provider

    if provider == "voyage":
        resp = _voyage_client().embed(
            texts, model=settings.voyage_embed_model, input_type=input_type
        )
        return resp.embeddings

    if provider == "chroma":
        embedder = _chroma_embedder()
        try:
            result = embedder(texts)
        except TypeError:
            result = embedder.embed_documents(texts)
        return _to_lists(result)

    # default: "local" sentence-transformers (e5 by default)
    prepared = _e5_prefix(texts, input_type)
    vecs = _local_model().encode(
        prepared, normalize_embeddings=True, batch_size=settings.embed_batch_size,
        show_progress_bar=False,
    )
    return _to_lists(vecs)


def embed_query(text: str) -> list[float]:
    return embed_texts([text], input_type="query")[0]
