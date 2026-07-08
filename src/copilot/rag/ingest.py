"""Ingest documents into a named collection (= agent slug).

Upgraded pipeline:
  - multi-format extraction (pdf/docx/pptx/txt/md/images) with OCR fallback
  - structure-aware chunking (section-coherent, page-tracked)
  - rich metadata: {topic, source, page_start, page_end, section, path, ocr}
    where `topic` is the immediate subfolder under the collection root
    (e.g. "02_CORE_Principle_and_Modular_Design").

Usage:
    python -m copilot.rag.ingest teaching ./data/teaching
    python -m copilot.rag.ingest teaching ./data/teaching --reset
    python -m copilot.rag.ingest teaching ./data/teaching --dry-run   # report only, no DB writes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg

from copilot.core.settings import settings
from copilot.rag.chunking import chunk_blocks
from copilot.rag.embeddings import embed_texts, vec_literal
from copilot.rag.extractors import SUPPORTED_EXTS, extract


def _topic_for(path: Path, root: Path) -> str:
    """First path component under the collection root = the topic tag."""
    try:
        rel = path.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else "_root"
    except ValueError:
        return "_root"


def ingest(collection: str, folder: str, reset: bool = False, dry_run: bool = False) -> None:
    root = Path(folder)
    if not root.exists():
        sys.exit(f"Folder not found: {root}")

    files = [
        p for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
        and ".ipynb_checkpoints" not in p.parts
        and p.name.lower() != "readme.md"
    ]
    if not files:
        sys.exit(f"No supported files under {root}")

    skipped = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() not in SUPPORTED_EXTS
        and ".ipynb_checkpoints" not in p.parts
    ]

    conn = None if dry_run else psycopg.connect(settings.database_url, autocommit=True)
    try:
        cur = conn.cursor() if conn else None
        if cur and reset:
            cur.execute("DELETE FROM documents WHERE collection = %s;", (collection,))
            print(f"• cleared collection '{collection}'")

        total_chunks = 0
        per_topic: dict[str, int] = {}
        ocr_files: list[str] = []
        empty_files: list[str] = []

        for f in files:
            topic = _topic_for(f, root)
            try:
                blocks = extract(f)
            except Exception as e:  # noqa: BLE001
                print(f"  ! extract failed: {f.name} ({e})")
                continue
            if not blocks:
                empty_files.append(str(f.relative_to(root)))
                continue
            if any(b.meta.get("ocr") for b in blocks):
                ocr_files.append(f.name)

            chunks = chunk_blocks(blocks, settings.chunk_tokens, settings.chunk_overlap)
            if not chunks:
                empty_files.append(str(f.relative_to(root)))
                continue

            if not dry_run:
                texts = [c.text for c in chunks]
                vectors: list[list[float]] = []
                for i in range(0, len(texts), 96):
                    vectors.extend(embed_texts(texts[i : i + 96], input_type="document"))
                rows = []
                for idx, (c, vec) in enumerate(zip(chunks, vectors)):
                    meta = {
                        "topic": topic,
                        "path": str(f),
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "section": c.section,
                    }
                    rows.append((collection, f.name, idx, c.text, json.dumps(meta), vec_literal(vec)))
                cur.executemany(
                    "INSERT INTO documents (collection, source, chunk_index, content, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s);",
                    rows,
                )

            total_chunks += len(chunks)
            per_topic[topic] = per_topic.get(topic, 0) + len(chunks)
            tag = " [OCR]" if any(b.meta.get("ocr") for b in blocks) else ""
            print(f"  ✓ [{topic}] {f.name}: {len(chunks)} chunks{tag}")
    finally:
        if conn:
            conn.close()

    print("\n" + ("— DRY RUN (no DB writes) —\n" if dry_run else ""))
    print(f"Files processed : {len(files)}")
    print(f"Total chunks    : {total_chunks}")
    print("Chunks per topic:")
    for t, n in sorted(per_topic.items()):
        print(f"    {t:45s} {n}")
    if ocr_files:
        print(f"OCR was used on {len(ocr_files)} file(s): {', '.join(ocr_files[:10])}"
              + (" …" if len(ocr_files) > 10 else ""))
    if empty_files:
        print(f"\n⚠ {len(empty_files)} file(s) yielded NO text (likely scanned w/o OCR stack, "
              f"or empty):")
        for e in empty_files[:20]:
            print(f"    {e}")
    if skipped:
        print(f"\n⚠ {len(skipped)} unsupported file(s) skipped: "
              + ", ".join(sorted({p.suffix.lower() for p in skipped})))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("collection", help="agent slug, e.g. 'teaching'")
    ap.add_argument("folder")
    ap.add_argument("--reset", action="store_true", help="clear collection before ingest")
    ap.add_argument("--dry-run", action="store_true", help="extract + chunk + report, no DB writes")
    args = ap.parse_args()
    ingest(args.collection, args.folder, reset=args.reset, dry_run=args.dry_run)
