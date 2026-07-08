"""HR Compliance Agent — STUB.

A working skeleton: RAG (`knowledge_search` on the 'hr_compliance' collection) and `web_search`
are wired, with a domain system prompt. Build out this agent by adding capability tools in
`tools.py` (see agents/teaching/tools.py for the pattern) and appending them in build_tools().
"""
from __future__ import annotations

from copilot.core.base_agent import BaseAgent
from copilot.core.registry import register
from copilot.tools.knowledge import make_knowledge_tool
from copilot.tools.web import web_search

# from .tools import CAPABILITY_TOOLS  # TODO: add domain tools here

SYSTEM_PROMPT = """You are the HR Compliance Agent. You support HR staff in reviewing documentation of external lecturers and freelancers.

RESPONSIBILITIES: analyze CVs and submitted documentation, detect missing information, and identify potential risk indicators for staff review.

PRINCIPLES: You ASSIST review; you do not make hiring or compliance decisions. Ground checks in the institution's HR policies via `knowledge_search`. Clearly separate 'missing/incomplete' from 'potential risk — needs human review'. Never assert a definitive compliance judgment; surface findings for an HR professional to decide. Handle personal data discreetly."""


@register
class HRComplianceAgent(BaseAgent):
    slug = "hr_compliance"
    name = "HR Compliance Agent"
    description = "Supports HR in reviewing documentation of external lecturers and freelancers."
    system_prompt = SYSTEM_PROMPT

    def build_tools(self):
        return [
            make_knowledge_tool(self.slug),
            web_search,
            # *CAPABILITY_TOOLS,  # TODO: add domain-specific capability tools
        ]
