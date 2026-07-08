"""Strategic Intelligence Agent — STUB.

A working skeleton: RAG (`knowledge_search` on the 'strategic' collection) and `web_search`
are wired, with a domain system prompt. Build out this agent by adding capability tools in
`tools.py` (see agents/teaching/tools.py for the pattern) and appending them in build_tools().
"""
from __future__ import annotations

from copilot.core.base_agent import BaseAgent
from copilot.core.registry import register
from copilot.tools.knowledge import make_knowledge_tool
from copilot.tools.web import web_search

# from .tools import CAPABILITY_TOOLS  # TODO: add domain tools here

SYSTEM_PROMPT = """You are the Strategic Intelligence Agent. You support university leadership with market and trend analysis.

RESPONSIBILITIES: monitor competitors, detect trends in higher education, and produce concise reports and dashboard-ready summaries.

PRINCIPLES: Use `web_search` for current external developments and `knowledge_search` for internal strategy context. Distinguish evidence from interpretation, and cite sources for external claims. Present balanced analysis with implications and options, not directives — leadership decides. Note uncertainty and data recency."""


@register
class StrategicIntelligenceAgent(BaseAgent):
    slug = "strategic"
    name = "Strategic Intelligence Agent"
    description = "Supports university leadership with market and trend analysis, competitor monitoring, and reports."
    system_prompt = SYSTEM_PROMPT

    def build_tools(self):
        return [
            make_knowledge_tool(self.slug),
            web_search,
            # *CAPABILITY_TOOLS,  # TODO: add domain-specific capability tools
        ]
