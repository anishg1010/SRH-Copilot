"""Full RAG: retrieve relevant chunks, then generate a grounded answer.

This is the "G" in RAG. Retrieval (retriever.py) finds the chunks; here we feed
them to a local open-source LLM (Ollama) with a prompt that forces the answer to
stay grounded in — and cite — the retrieved sources.

    python -m copilot.rag.generate teaching "How does the CORE principle work?"
"""
from __future__ import annotations

import argparse

from copilot.core.config import config
from copilot.core.llm import LLMUnavailable, generate as llm_generate
from copilot.rag.retriever import Passage, retrieve

SYSTEM = (
    "You are a teaching-support assistant for university lecturers. Answer ONLY using "
    "the provided context passages. If the context does not contain the answer, say so "
    "plainly. Cite sources inline as [n] using the passage numbers. Be concise and factual. "
    "Answer in the same language as the question."
)


def _build_prompt(question: str, passages: list[Passage]) -> str:
    context = "\n\n".join(
        f"[{i+1}] (source: {p.source}"
        + (f", topic: {p.topic}" if p.topic else "")
        + (f", p.{p.page_start}" if p.page_start else "")
        + f")\n{p.content}"
        for i, p in enumerate(passages)
    )
    return (
        f"CONTEXT PASSAGES:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the context above, citing sources as [n]:"
    )


def answer(
    collection: str,
    question: str,
    k: int | None = None,
    topics: list[str] | None = None,
) -> dict:
    """Retrieve then generate. Returns {answer, passages}."""
    passages = retrieve(
        collection, question, k=k or config.retrieve_k,
        topics=topics, min_score=config.min_score or None,
    )
    if not passages:
        return {"answer": "No relevant passages found in the knowledge base.", "passages": []}

    prompt = _build_prompt(question, passages)
    try:
        text = llm_generate(prompt, system=SYSTEM)
    except LLMUnavailable as e:
        text = (
            f"[LLM not running — retrieval succeeded, generation skipped]\n{e}\n\n"
            "Retrieved passages are listed below; start Ollama to get a written answer."
        )
    return {"answer": text, "passages": passages}


def _format(result: dict) -> str:
    out = [result["answer"], "", "─" * 60, "SOURCES:"]
    for i, p in enumerate(result["passages"]):
        loc = []
        if p.topic:
            loc.append(p.topic)
        if p.page_start:
            loc.append(f"p.{p.page_start}")
        out.append(f"  [{i+1}] {p.source}" + (f" ({' · '.join(loc)})" if loc else "")
                   + f"  — relevance {p.score:.3f}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("collection")
    ap.add_argument("question")
    ap.add_argument("--topics", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()
    print(_format(answer(args.collection, args.question, k=args.k, topics=args.topics)))
