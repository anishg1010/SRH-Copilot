"""Chroma-based collection-scoped retrieval with metadata + optional topic filter."""
from __future__ import annotations

from dataclasses import dataclass

from copilot.core.config import config as settings
from copilot.rag.embeddings import embed_query
from copilot.rag.store import get_collection


@dataclass
class Passage:
    source: str
    content: str
    score: float
    topic: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


def retrieve(
    collection: str,
    query: str,
    k: int | None = None,
    topics: list[str] | None = None,
    min_score: float | None = None,
) -> list[Passage]:
    """Return the k most relevant passages within a Chroma collection.

    Args:
        collection: agent slug.
        query: search text.
        k: number of results (defaults to settings.retrieve_k).
        topics: restrict to these metadata.topic values (folder names).
        min_score: drop passages below this cosine similarity (0..1).
    """
    k = k or settings.retrieve_k
    col = get_collection(collection)
    qvec = embed_query(query)

    where = {"topic": {"$in": topics}} if topics else None
    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    passages: list[Passage] = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1.0 - float(dist)   # cosine distance → similarity
        if min_score is not None and score < min_score:
            continue
        meta = meta or {}
        passages.append(Passage(
            source=meta.get("source", "?"),
            content=doc,
            score=score,
            topic=meta.get("topic"),
            page_start=meta.get("page_start"),
            page_end=meta.get("page_end"),
            section=meta.get("section"),
        ))
    return passages


def list_topics(collection: str) -> list[tuple[str, int]]:
    """Return (topic, chunk_count) pairs by scanning the collection's metadata."""
    col = get_collection(collection)
    got = col.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for m in got.get("metadatas") or []:
        t = (m or {}).get("topic", "(none)")
        counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items())


def format_passages(passages: list[Passage]) -> str:
    if not passages:
        return "No relevant passages found in the knowledge base."
    out = []
    for i, p in enumerate(passages):
        loc = []
        if p.topic:
            loc.append(p.topic)
        if p.page_start:
            loc.append(f"p.{p.page_start}" + (f"-{p.page_end}" if p.page_end and p.page_end != p.page_start else ""))
        if p.section:
            loc.append(f"§ {p.section}")
        locstr = " · ".join(loc)
        out.append(
            f"[{i+1}] {p.source}" + (f" ({locstr})" if locstr else "")
            + f" — relevance {p.score:.3f}\n{p.content[:500]}"
            + ("…" if len(p.content) > 500 else "")
        )
    return "\n\n".join(out)
