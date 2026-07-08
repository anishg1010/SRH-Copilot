"""Structure-aware chunking.

Instead of a blind sliding window over the whole document, we group extractor
`Block`s into chunks that:
  - never merge across a section-heading boundary (a chunk stays within one section),
  - stay under `chunk_tokens` (splitting a long section with `chunk_overlap`),
  - carry provenance: the source page range and section heading.

This keeps a chunk a coherent unit (a full competence definition, one rubric,
a slide) and lets retrieval cite "source, p. N (section: …)".

The tuning knobs (chunk_tokens, chunk_overlap) stay in settings so you can
experiment without touching this logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from copilot.rag.extractors import Block

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    page_start: int | None
    page_end: int | None
    section: str | None


def _ntok(text: str) -> int:
    return len(_enc.encode(text))


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    tokens = _enc.encode(text)
    if len(tokens) <= size:
        return [text]
    out, start = [], 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        out.append(_enc.decode(tokens[start:end]).strip())
        if end == len(tokens):
            break
        start += size - overlap
    return [c for c in out if c]


def chunk_blocks(blocks: list[Block], size: int, overlap: int) -> list[Chunk]:
    """Group blocks into section-coherent, size-limited chunks."""
    chunks: list[Chunk] = []
    buf: list[Block] = []
    buf_tokens = 0

    def flush():
        nonlocal buf, buf_tokens
        if not buf:
            return
        section = next((b.section for b in buf if b.section), None)
        pages = [b.page for b in buf if b.page is not None]
        page_start = min(pages) if pages else None
        page_end = max(pages) if pages else None
        joined = "\n".join(b.text for b in buf).strip()
        for piece in _split_long(joined, size, overlap):
            chunks.append(Chunk(piece, page_start, page_end, section))
        buf, buf_tokens = [], 0

    prev_section = object()  # sentinel
    for b in blocks:
        # New section heading → close the current chunk so sections don't bleed.
        if b.meta.get("heading") and b.section != prev_section:
            flush()
            prev_section = b.section

        t = _ntok(b.text)
        # A single oversized block: flush what we have, emit it split on its own.
        if t > size:
            flush()
            for piece in _split_long(b.text, size, overlap):
                chunks.append(Chunk(piece, b.page, b.page, b.section))
            continue

        if buf_tokens + t > size:
            flush()
        buf.append(b)
        buf_tokens += t

    flush()
    return [c for c in chunks if c.text]
