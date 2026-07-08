"""Student Service Agent — STUB.

A working skeleton: RAG (`knowledge_search` on the 'student_services' collection) and `web_search`
are wired, with a domain system prompt. Build out this agent by adding capability tools in
`tools.py` (see agents/teaching/tools.py for the pattern) and appending them in build_tools().
"""
from __future__ import annotations

from copilot.core.base_agent import BaseAgent
from copilot.core.registry import register
from copilot.tools.knowledge import make_knowledge_tool
from copilot.tools.web import web_search

# from .tools import CAPABILITY_TOOLS  # TODO: add domain tools here

SYSTEM_PROMPT = """You are the Student Services Agent. You help students find information and navigate university services.

RESPONSIBILITIES: answer FAQs, identify the student's underlying need, route requests to the correct department, and provide links to relevant services.

PRINCIPLES: Ground answers in official university information via `knowledge_search` before responding. If a request needs a human or a specific office, say which department and how to reach it rather than guessing. Be concise, friendly, and accurate; never invent policies or deadlines."""


@register
class StudentServicesAgent(BaseAgent):
    slug = "student_services"
    name = "Student Service Agent"
    description = "Helps students find information and navigates university services; routes requests to the right department."
    system_prompt = SYSTEM_PROMPT

    def build_tools(self):
        return [
            make_knowledge_tool(self.slug),
            web_search,
            # *CAPABILITY_TOOLS,  # TODO: add domain-specific capability tools
        ]
