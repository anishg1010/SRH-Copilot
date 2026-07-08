"""Shared web search tool (Tavily default; swap the request body to change providers)."""
from __future__ import annotations

import httpx
from langchain_core.tools import tool

from copilot.core.settings import settings


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current, external information not in the knowledge base.

    Args:
        query: The search query.
        max_results: How many results (default 5).
    """
    if not settings.tavily_api_key:
        return "Web search unavailable: TAVILY_API_KEY not configured."
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return f"Web search failed: {e}"
    results = data.get("results", [])
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"• {r.get('title', 'Untitled')}\n  {r.get('url', '')}\n  {r.get('content', '')[:400]}"
        for r in results
    )
