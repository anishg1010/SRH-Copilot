"""ChromaDB vector store (local, persistent)."""
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
    return _client().get_or_create_collection(
        name=_collection_name(collection),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(collection: str) -> None:
    name = _collection_name(collection)
    try:
        _client().delete_collection(name)
    except Exception:
        pass


def _clean_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool metadata values.
    Drop None, coerce anything else to string."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, (str, int, float)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def add_chunks(collection, ids, documents, embeddings, metadatas):
    col = get_collection(collection)
    metadatas = [_clean_metadata(m) for m in metadatas]
    col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def count(collection: str) -> int:
    return get_collection(collection).count()
