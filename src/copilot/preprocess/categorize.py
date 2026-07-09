"""Step 2 of pre-processing: understand + categorize documents by the TEACHING
AGENT's functions (from the project brief), ignoring the old 01/02/03 folders.

Categories = the Teaching Innovation Agent (CORE Bot) functions:
  learning_objectives   — generate learning objectives
  course_module_design  — support course & module design
  teaching_methods      — suggest teaching methods
  assessment_rubrics    — design assessments & rubrics
  responsible_ai        — integrate AI into teaching responsibly
  general_core          — CORE framework / general (fits none strongly)

Method (unsupervised + anchored):
  1. Embed each document (mean of its chunk embeddings).
  2. KMeans-cluster the docs to find natural groupings.
  3. For each doc, also score cosine similarity to each function's anchor text,
     and assign the best-matching function as the proposed category.
     (Clusters are reported too, so you can see natural groupings alongside.)
  4. Write an editable mapping CSV: source, proposed_category, cluster, scores...

    python -m copilot.preprocess.categorize --doc-json doc_json --k 6
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# The 5 teaching-agent functions, as anchor descriptions for similarity scoring.
FUNCTION_ANCHORS = {
    "learning_objectives": (
        "Formulating and generating competence-oriented, measurable learning objectives "
        "and outcomes. Lernziele, Kompetenzen, Taxonomie, Bloom, SOLO, learning outcomes."
    ),
    "course_module_design": (
        "Designing courses and modules, curriculum and module structure, module descriptors, "
        "study programme design. Modulkonzeption, Studiengang, CORE modular design, curriculum."
    ),
    "teaching_methods": (
        "Teaching and active-learning methods, didactics, instructional design, classroom "
        "activities, icebreakers. Methoden, Didaktik, aktivierende Methoden, Lehre, instruction."
    ),
    "assessment_rubrics": (
        "Designing assessments, exams, and grading rubrics with criteria. Pruefung, "
        "Pruefungsformate, Bewertung, Rubric, assessment, evaluation criteria."
    ),
    "responsible_ai": (
        "Responsible and ethical use of generative AI in teaching and writing, AI guidelines, "
        "AI literacy. Generative KI, KI in der Lehre, AI ethics, ChatGPT, writing with AI."
    ),
    "general_core": (
        "General CORE principle, future skills, higher-education strategy, administration, "
        "templates, regulations. CORE Prinzip, Future Skills, Ordnung, Vorlage, strategy."
    ),
}


def _load_docs(doc_json_dir: Path):
    docs = []
    for p in sorted(doc_json_dir.glob("*.json")):
        if p.name == "_index.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        docs.append(d)
    return docs


def categorize(doc_json_dir: str, k: int = 6, out_csv: str = "categorization_mapping.csv") -> None:
    import numpy as np
    from sklearn.cluster import KMeans

    from copilot.rag.embeddings import embed_texts

    ddir = Path(doc_json_dir)
    docs = _load_docs(ddir)
    if not docs:
        raise SystemExit(f"No document JSON in {ddir}. Run doc_json first.")

    # 1. embed each document — use a generous text window (title + preview + start of body)
    doc_texts = [
        (d["source"] + "\n" + d.get("preview", "") + "\n" + d.get("full_text", "")[:2000])
        for d in docs
    ]
    print(f"Embedding {len(docs)} documents…")
    doc_vecs = np.array(embed_texts(doc_texts, input_type="document"), dtype=float)

    # 2. anchor embeddings for the functions
    anchor_names = list(FUNCTION_ANCHORS.keys())
    anchor_vecs = np.array(
        embed_texts(list(FUNCTION_ANCHORS.values()), input_type="query"), dtype=float
    )

    # normalize for cosine
    def _norm(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1
        return m / n
    dv, av = _norm(doc_vecs), _norm(anchor_vecs)
    sims = dv @ av.T                       # (num_docs, num_functions) cosine similarities

    # 3. unsupervised clusters (for reference alongside the anchored label)
    k = min(k, len(docs))
    clusters = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(doc_vecs)

    # 4. write mapping CSV
    rows = []
    for i, d in enumerate(docs):
        best = int(np.argmax(sims[i]))
        proposed = anchor_names[best]
        row = {
            "source": d["source"],
            "current_folder": d.get("current_folder", ""),
            "proposed_category": proposed,
            "confidence": round(float(sims[i][best]), 3),
            "cluster": int(clusters[i]),
        }
        for j, name in enumerate(anchor_names):
            row[f"score_{name}"] = round(float(sims[i][j]), 3)
        rows.append(row)

    rows.sort(key=lambda r: (r["proposed_category"], -r["confidence"]))
    fieldnames = (["source", "current_folder", "proposed_category", "confidence", "cluster"]
                  + [f"score_{n}" for n in anchor_names])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # summary
    from collections import Counter
    counts = Counter(r["proposed_category"] for r in rows)
    print(f"\n✓ Mapping written to {out_csv}")
    print("Proposed category distribution:")
    for cat, n in counts.most_common():
        print(f"    {cat:22s} {n}")
    print("\nEdit the 'proposed_category' column as needed, then run:")
    print(f"    python -m copilot.preprocess.apply_mapping {out_csv} --dest data_recategorized")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-json", default="doc_json")
    ap.add_argument("--k", type=int, default=6, help="number of KMeans clusters")
    ap.add_argument("--out", default="categorization_mapping.csv")
    args = ap.parse_args()
    categorize(args.doc_json, k=args.k, out_csv=args.out)
