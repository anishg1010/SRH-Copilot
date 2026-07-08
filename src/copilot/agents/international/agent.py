"""International Office Agent — STUB.

A working skeleton: RAG (`knowledge_search` on the 'international' collection) and `web_search`
are wired, with a domain system prompt. Build out this agent by adding capability tools in
`tools.py` (see agents/teaching/tools.py for the pattern) and appending them in build_tools().
"""
from __future__ import annotations

from copilot.core.base_agent import BaseAgent
from copilot.core.registry import register
from copilot.tools.knowledge import make_knowledge_tool
from copilot.tools.web import web_search

# from .tools import CAPABILITY_TOOLS  # TODO: add domain tools here

SYSTEM_PROMPT = """You are the International Office Agent. You support international students and staff.

RESPONSIBILITIES: answer questions on admissions for international applicants, mobility/exchange, and cross-border administrative processes; route complex cases (e.g. visa, legal) to the right office.

PRINCIPLES: Ground answers in official international-office information via `knowledge_search`. For legal/visa specifics, provide general guidance and direct the person to the responsible office and official sources — never give definitive legal advice. Be culturally sensitive and clear, mindful users may be non-native speakers."""


@register
class InternationalOfficeAgent(BaseAgent):
    slug = "international"
    name = "International Office Agent"
    description = "Supports international students and staff with mobility, visa guidance routing, and cross-border processes."
    system_prompt = SYSTEM_PROMPT

    def build_tools(self):
        return [
            make_knowledge_tool(self.slug),
            web_search,
            # *CAPABILITY_TOOLS,  # TODO: add domain-specific capability tools
        ]
