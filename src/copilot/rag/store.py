"""ChromaDB vector store (local, persistent).

Replaces the pgvector store. Chroma persists to `settings.chroma_dir` on disk —
no server, no Docker, no network. Each agent/collection maps to a Chroma
collection named `{prefix}_{collection}`.

We pass our OWN precomputed embeddings (from copilot.rag.embeddings), so Chroma
is used purely as the vector index + metadata store — keeping the embedding
provider swappable independently of Chroma.
"""
from __future__ import annotations

from functools import lru_cache

from copilot.core.config import config as settings


@lru_cache(maxsize=1)
def _client():
    import chromadb
    return chromadb.PersistentClient(path=settings.chroma_dir)


def _collection_name(collection: str) -> str:
    return f"{settings.chroma_collection_prefix}_{collection}"


def get_collection(collection: str):
    """Get or create the Chroma collection for an agent slug (cosine space)."""
    return _client().get_or_create_collection(
        name=_collection_name(collection),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(collection: str) -> None:
    """Delete the collection if it exists (used by --reset)."""
    name = _collection_name(collection)
    try:
        _client().delete_collection(name)
    except Exception:
        pass  # didn't exist


def add_chunks(
    collection: str,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    col = get_collection(collection)
    col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def count(collection: str) -> int:
    return get_collection(collection).count()
