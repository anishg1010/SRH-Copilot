"""Ingest every collection's documents in one pass.

    python -m copilot.rag.ingest_all
    python -m copilot.rag.ingest_all --reset

Iterates over registered agents and ingests ./data/<slug> if it contains documents.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import copilot.agents  # noqa: F401  (registers agents so we know the slugs)
from copilot.core.registry import all_agents
from copilot.rag.ingest import ingest

DATA_ROOT = Path("data")


def main(reset: bool = False) -> None:
    for agent in all_agents():
        folder = DATA_ROOT / agent.slug
        docs = [
            p for p in folder.rglob("*")
            if p.suffix.lower() in {".txt", ".md", ".pdf"} and p.name.lower() != "readme.md"
        ]
        if not docs:
            print(f"– {agent.slug}: no documents, skipping")
            continue
        print(f"→ {agent.slug}:")
        ingest(agent.slug, str(folder), reset=reset)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    main(reset=args.reset)
