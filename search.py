#!/usr/bin/env python3
"""Usage: python search.py "Welche Regeln gelten für KI-Software?" """
import json, sys, numpy as np
from sentence_transformers import SentenceTransformer

chunks = [json.loads(l) for l in open("output/chunks.jsonl")]
vecs   = np.load("output/vectors.npy")
meta   = json.load(open("output/embed_meta.json"))
m      = SentenceTransformer(meta["model"])

q = m.encode(f"query: {' '.join(sys.argv[1:])}", normalize_embeddings=True)  # "query:" NOT "passage:"
scores = vecs @ q
for i in np.argsort(-scores)[:5]:
    c = chunks[i]
    print(f"\n[{scores[i]:.3f}] {c['source_file']} p{c['page']} ({c['language']}, ocr={c['is_ocr']})")
    print(c["text"][:320].replace("\n", " "))