#!/usr/bin/env python3
"""Embed chunks.jsonl -> vectors.npy. Separate from chunking on purpose:
swapping the embedding model must not force a re-chunk."""
import json, sys, numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-large"   # 1024-dim. Use -base (768) if too slow.

chunks = [json.loads(l) for l in open("output/chunks.jsonl")]
print(f"{len(chunks)} chunks", file=sys.stderr)

m = SentenceTransformer(MODEL)
vecs = m.encode(
    [f"passage: {c['embed_text']}" for c in chunks],   # E5 REQUIRES the prefix
    normalize_embeddings=True,                          # cosine == dot product
    batch_size=16, show_progress_bar=True,
).astype("float32")

np.save("output/vectors.npy", vecs)
json.dump([c["chunk_id"] for c in chunks], open("output/chunk_ids.json", "w"))
json.dump({"model": MODEL, "dim": int(vecs.shape[1]), "n": len(chunks)},
          open("output/embed_meta.json", "w"), indent=2)
print(f"{vecs.shape} -> output/vectors.npy")