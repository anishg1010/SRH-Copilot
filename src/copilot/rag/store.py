"""Initialize Postgres: pgvector extension + a single shared documents table.

Every agent's documents live in one table, partitioned logically by a `collection`
column (= the agent slug). Retrieval always filters by collection, so agents never
see each other's knowledge. One table keeps ops simple; the index scales fine and
you can PARTITION BY LIST(collection) later if a collection grows very large.

LangGraph checkpoint tables are created separately by PostgresSaver.setup().
"""
from __future__ import annotations

import psycopg

from copilot.core.settings import settings


def init_db() -> None:
    with psycopg.connect(settings.database_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id          BIGSERIAL PRIMARY KEY,
                collection  TEXT NOT NULL,
                source      TEXT NOT NULL,
                chunk_index INT  NOT NULL,
                content     TEXT NOT NULL,
                metadata    JSONB DEFAULT '{{}}'::jsonb,
                embedding   VECTOR({settings.embed_dim}) NOT NULL
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS documents_embedding_idx "
            "ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS documents_collection_idx ON documents (collection);"
        )
        # GIN index on metadata so topic filters (metadata->>'topic') stay fast.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS documents_metadata_idx "
            "ON documents USING gin (metadata);"
        )
    print("✓ pgvector enabled, shared `documents` table + indexes ready.")


if __name__ == "__main__":
    init_db()
