"""Ingest documents into a named collection (= agent slug).

Usage:
    python -m copilot.rag.ingest teaching ./data/teaching
    python -m copilot.rag.ingest feedback ./data/feedback --reset

Supports .txt, .md, .pdf.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg
import tiktoken
from pypdf import PdfReader

from copilot.core.settings import settings
from copilot.rag.embeddings import embed_texts, vec_literal

_enc = tiktoken.get_encoding("cl100k_base")


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    tokens = _enc.encode(text)
    if not tokens:
        return []
    out, start = [], 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        out.append(_enc.decode(tokens[start:end]).strip())
        if end == len(tokens):
            break
        start += size - overlap
    return [c for c in out if c]


def ingest(collection: str, folder: str, reset: bool = False) -> None:
    docs_dir = Path(folder)
    if not docs_dir.exists():
        sys.exit(f"Folder not found: {docs_dir}")
    files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
    if not files:
        sys.exit(f"No .txt/.md/.pdf under {docs_dir}")

    with psycopg.connect(settings.database_url, autocommit=True) as conn, conn.cursor() as cur:
        if reset:
            cur.execute("DELETE FROM documents WHERE collection = %s;", (collection,))
            print(f"• cleared collection '{collection}'")
        total = 0
        for f in files:
            chunks = _chunk(_read(f), settings.chunk_tokens, settings.chunk_overlap)
            if not chunks:
                print(f"  ! skipped (no text): {f.name}")
                continue
            vectors: list[list[float]] = []
            for i in range(0, len(chunks), 96):
                vectors.extend(embed_texts(chunks[i : i + 96], input_type="document"))
            rows = [
                (collection, f.name, idx, chunk, json.dumps({"path": str(f)}), vec_literal(vec))
                for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
            ]
            cur.executemany(
                "INSERT INTO documents (collection, source, chunk_index, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s);",
                rows,
            )
            total += len(rows)
            print(f"  ✓ {f.name}: {len(rows)} chunks")
    print(f"\n✓ Ingested {total} chunks into '{collection}' from {len(files)} file(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("collection", help="agent slug, e.g. 'teaching'")
    ap.add_argument("folder")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    ingest(args.collection, args.folder, reset=args.reset)
