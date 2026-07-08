"""RAG inspection helpers for tuning.

    python -m copilot.rag.inspect topics teaching
        → list topics and chunk counts in a collection

    python -m copilot.rag.inspect query teaching "competence-oriented assessment" 
        → run a retrieval and print the ranked passages with provenance + scores

    python -m copilot.rag.inspect query teaching "..." --topics 02_CORE_Principle_and_Modular_Design --k 8

Use this to eyeball whether chunking/thresholds are returning the right sources
before wiring anything into the agent.
"""
from __future__ import annotations

import argparse

from copilot.rag.retriever import format_passages, list_topics, retrieve


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("topics")
    t.add_argument("collection")

    q = sub.add_parser("query")
    q.add_argument("collection")
    q.add_argument("text")
    q.add_argument("--topics", nargs="*", default=None)
    q.add_argument("--k", type=int, default=6)
    q.add_argument("--min-score", type=float, default=None)

    args = ap.parse_args()

    if args.cmd == "topics":
        rows = list_topics(args.collection)
        total = sum(n for _, n in rows)
        print(f"Collection '{args.collection}' — {total} chunks across {len(rows)} topics:\n")
        for topic, n in rows:
            print(f"  {topic or '(none)':45s} {n}")
    elif args.cmd == "query":
        passages = retrieve(
            args.collection, args.text,
            k=args.k, topics=args.topics, min_score=args.min_score,
        )
        print(f"Query: {args.text!r}  (k={args.k}, topics={args.topics})\n")
        print(format_passages(passages))


if __name__ == "__main__":
    main()
