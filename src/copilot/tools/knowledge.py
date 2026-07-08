"""Factory for a collection-scoped knowledge-search tool.

Each agent calls `make_knowledge_tool(slug)` to get a `knowledge_search` tool wired
to its own document collection. This is why every agent gets RAG for free.
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool

from copilot.rag.retriever import format_passages, retrieve


def make_knowledge_tool(collection: str) -> StructuredTool:
    def knowledge_search(query: str) -> str:
        """Search this agent's institutional knowledge base for grounding passages.
        Use FIRST for anything specific to the university's policies, templates,
        or domain documents."""
        return format_passages(retrieve(collection, query))

    return StructuredTool.from_function(
        func=knowledge_search,
        name="knowledge_search",
        description=(
            "Search the institution's knowledge base relevant to this agent's domain. "
            "Returns grounding passages. Use before answering domain-specific questions."
        ),
    )
