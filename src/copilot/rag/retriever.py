"""Collection-scoped similarity search with metadata + optional topic filter."""
from __future__ import annotations

from dataclasses import dataclass

import psycopg

from copilot.core.settings import settings
from copilot.rag.embeddings import embed_query, vec_literal


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
    """Return the k most relevant passages within a collection.

    Args:
        collection: agent slug.
        query: the search text.
        k: number of results (defaults to settings.retrieve_k).
        topics: if given, restrict to these metadata.topic values (folder names).
        min_score: if given, drop passages below this cosine similarity (0..1).
    """
    k = k or settings.retrieve_k
    qvec = vec_literal(embed_query(query))

    where = ["collection = %s"]
    params: list = [qvec, collection]  # qvec first (used in SELECT), then collection
    if topics:
        where.append("metadata->>'topic' = ANY(%s)")
        params.append(topics)
    params.append(qvec)  # ORDER BY vector
    params.append(k)

    sql = f"""
        SELECT source, content, 1 - (embedding <=> %s::vector) AS score,
               metadata->>'topic'      AS topic,
               (metadata->>'page_start')::int AS page_start,
               (metadata->>'page_end')::int   AS page_end,
               metadata->>'section'    AS section
        FROM documents
        WHERE {' AND '.join(where)}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    passages = [
        Passage(source=s, content=c, score=float(sc), topic=tp,
                page_start=ps, page_end=pe, section=sec)
        for s, c, sc, tp, ps, pe, sec in rows
    ]
    if min_score is not None:
        passages = [p for p in passages if p.score >= min_score]
    return passages


def list_topics(collection: str) -> list[tuple[str, int]]:
    """Return (topic, chunk_count) pairs for a collection — handy for tuning."""
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT metadata->>'topic' AS topic, COUNT(*) "
            "FROM documents WHERE collection = %s GROUP BY topic ORDER BY topic;",
            (collection,),
        )
        return [(t, n) for t, n in cur.fetchall()]


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
            + f" — relevance {p.score:.2f}\n{p.content}"
        )
    return "\n\n".join(out)
