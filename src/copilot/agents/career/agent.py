"""Career Support Agent — STUB.

A working skeleton: RAG (`knowledge_search` on the 'career' collection) and `web_search`
are wired, with a domain system prompt. Build out this agent by adding capability tools in
`tools.py` (see agents/teaching/tools.py for the pattern) and appending them in build_tools().
"""
from __future__ import annotations

from copilot.core.base_agent import BaseAgent
from copilot.core.registry import register
from copilot.tools.knowledge import make_knowledge_tool
from copilot.tools.web import web_search

# from .tools import CAPABILITY_TOOLS  # TODO: add domain tools here

SYSTEM_PROMPT = """You are the Career Support Agent. You help students improve job application documents.

RESPONSIBILITIES: analyze CVs and cover letters, detect common errors, suggest concrete improvements, and prepare students before career consultations.

PRINCIPLES: Give specific, actionable feedback tied to the target role where known. Ground advice in the university's careers guidance via `knowledge_search`. Preserve the student's authentic voice — suggest, don't rewrite wholesale. Be encouraging but honest about weaknesses."""


@register
class CareerSupportAgent(BaseAgent):
    slug = "career"
    name = "Career Support Agent"
    description = "Helps students improve CVs and cover letters and prepares them before career consultations."
    system_prompt = SYSTEM_PROMPT

    def build_tools(self):
        return [
            make_knowledge_tool(self.slug),
            web_search,
            # *CAPABILITY_TOOLS,  # TODO: add domain-specific capability tools
        ]
