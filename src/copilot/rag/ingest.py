"""Ingest documents into a ChromaDB collection, with JSON artifacts.

Produces THREE things:
  1. artifacts/<collection>_extracted.json   → raw text BEFORE chunking
         (one entry per source file: its blocks with page/section provenance)
  2. artifacts/<collection>_chunks.json      → the chunks AFTER chunking
         (each chunk with its text + metadata: topic, source, page range, section)
  3. the Chroma collection, populated with embeddings for retrieval

Usage:
    python -m copilot.rag.ingest teaching ./data/teaching
    python -m copilot.rag.ingest teaching ./data/teaching --reset
    python -m copilot.rag.ingest teaching ./data/teaching --no-embed   # JSON only, skip Chroma
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from copilot.core.config import config as settings
from copilot.rag.chunking import chunk_blocks
from copilot.rag.extractors import SUPPORTED_EXTS, extract

ARTIFACT_DIR = Path("artifacts")


def _topic_for(path: Path, root: Path) -> str:
    """The immediate parent folder name is the topic tag.

    Works regardless of nesting depth, e.g. both
      data/teaching/<Topic>/file.pdf
      data/teaching/structured/<Topic>/file.pdf
    yield <Topic>. Files directly in the root have topic '_root'.
    """
    parent = path.parent
    if parent == root:
        return "_root"
    return parent.name


def ingest(collection: str, folder: str, reset: bool = False, no_embed: bool = False) -> None:
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

    ARTIFACT_DIR.mkdir(exist_ok=True)

    extracted_json: list[dict] = []   # BEFORE chunking
    chunks_json: list[dict] = []      # AFTER chunking
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

        # --- artifact 1: raw extraction (before chunking) ---
        extracted_json.append({
            "source": f.name,
            "path": str(f),
            "topic": topic,
            "num_blocks": len(blocks),
            "full_text": "\n".join(b.text for b in blocks),
            "blocks": [
                {"text": b.text, "page": b.page, "section": b.section, "meta": b.meta}
                for b in blocks
            ],
        })

        # --- chunk ---
        chunks = chunk_blocks(blocks, settings.chunk_tokens, settings.chunk_overlap)
        if not chunks:
            empty_files.append(str(f.relative_to(root)))
            continue

        for idx, c in enumerate(chunks):
            chunks_json.append({
                "id": f"{f.name}::{idx}",
                "text": c.text,
                "metadata": {
                    "collection": collection,
                    "source": f.name,
                    "topic": topic,
                    "chunk_index": idx,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section": c.section,
                    "path": str(f),
                },
            })
        per_topic[topic] = per_topic.get(topic, 0) + len(chunks)
        tag = " [OCR]" if any(b.meta.get("ocr") for b in blocks) else ""
        print(f"  ✓ [{topic}] {f.name}: {len(blocks)} blocks → {len(chunks)} chunks{tag}")

    # --- write JSON artifacts ---
    extracted_path = ARTIFACT_DIR / f"{collection}_extracted.json"
    chunks_path = ARTIFACT_DIR / f"{collection}_chunks.json"
    extracted_path.write_text(json.dumps(extracted_json, ensure_ascii=False, indent=2), encoding="utf-8")
    chunks_path.write_text(json.dumps(chunks_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- load into Chroma (unless --no-embed) ---
    if not no_embed and chunks_json:
        from copilot.rag.embeddings import embed_texts
        from copilot.rag.store import add_chunks, count, get_collection, reset_collection

        if reset:
            reset_collection(collection)
        get_collection(collection)  # ensure exists

        texts = [c["text"] for c in chunks_json]
        ids = [c["id"] for c in chunks_json]
        metas = [c["metadata"] for c in chunks_json]

        print(f"\nEmbedding {len(texts)} chunks with provider '{settings.embed_provider}'…")
        BATCH = 128
        for i in range(0, len(texts), BATCH):
            vecs = embed_texts(texts[i : i + BATCH], input_type="document")
            add_chunks(
                collection,
                ids[i : i + BATCH],
                texts[i : i + BATCH],
                vecs,
                metas[i : i + BATCH],
            )
            print(f"    stored {min(i + BATCH, len(texts))}/{len(texts)}")
        print(f"✓ Chroma collection now holds {count(collection)} chunks.")

    # --- report ---
    print("\n" + "=" * 60)
    print(f"Files processed : {len(files)}")
    print(f"Total chunks    : {len(chunks_json)}")
    print("Chunks per topic:")
    for t, n in sorted(per_topic.items()):
        print(f"    {t:45s} {n}")
    if ocr_files:
        print(f"OCR used on {len(ocr_files)} file(s): {', '.join(ocr_files[:10])}"
              + (" …" if len(ocr_files) > 10 else ""))
    if empty_files:
        print(f"\n⚠ {len(empty_files)} file(s) yielded NO text:")
        for e in empty_files[:20]:
            print(f"    {e}")
    print("\nJSON artifacts written:")
    print(f"    {extracted_path}   (before chunking)")
    print(f"    {chunks_path}   (after chunking)")
    if no_embed:
        print("\n(--no-embed: skipped Chroma load; JSON only)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("collection", help="agent slug, e.g. 'teaching'")
    ap.add_argument("folder")
    ap.add_argument("--reset", action="store_true", help="clear the Chroma collection first")
    ap.add_argument("--no-embed", action="store_true", help="write JSON artifacts only, skip Chroma")
    args = ap.parse_args()
    ingest(args.collection, args.folder, reset=args.reset, no_embed=args.no_embed)
