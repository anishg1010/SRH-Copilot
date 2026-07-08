"""Collection-scoped similarity search."""
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


def retrieve(collection: str, query: str, k: int | None = None) -> list[Passage]:
    """Return the k most relevant passages within a single collection."""
    k = k or settings.retrieve_k
    qvec = vec_literal(embed_query(query))
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT source, content, 1 - (embedding <=> %s::vector) AS score
            FROM documents
            WHERE collection = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (qvec, collection, qvec, k),
        )
        return [Passage(s, c, float(sc)) for s, c, sc in cur.fetchall()]


def format_passages(passages: list[Passage]) -> str:
    if not passages:
        return "No relevant passages found in the knowledge base."
    return "\n\n".join(
        f"[{i+1}] (source: {p.source}, relevance: {p.score:.2f})\n{p.content}"
        for i, p in enumerate(passages)
    )
