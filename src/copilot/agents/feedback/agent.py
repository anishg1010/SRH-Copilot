"""Feedback Analytics Agent — STUB.

A working skeleton: RAG (`knowledge_search` on the 'feedback' collection) and `web_search`
are wired, with a domain system prompt. Build out this agent by adding capability tools in
`tools.py` (see agents/teaching/tools.py for the pattern) and appending them in build_tools().
"""
from __future__ import annotations

from copilot.core.base_agent import BaseAgent
from copilot.core.registry import register
from copilot.tools.knowledge import make_knowledge_tool
from copilot.tools.web import web_search

# from .tools import CAPABILITY_TOOLS  # TODO: add domain tools here

SYSTEM_PROMPT = """You are the Feedback Analytics Agent. You analyze student feedback and survey comments at scale.

RESPONSIBILITIES: cluster comments into themes, assess sentiment, detect recurring issues, and generate summary reports for staff.

PRINCIPLES: Base findings only on the provided feedback data; do not fabricate quotes or trends. Separate observation (what the data shows) from recommendation (what staff might do). Present themes with rough frequency and representative (paraphrased) examples. Flag low-confidence conclusions when the sample is small."""


@register
class FeedbackAnalyticsAgent(BaseAgent):
    slug = "feedback"
    name = "Feedback Analytics Agent"
    description = "Analyzes large volumes of student feedback and survey comments: clustering, sentiment, recurring issues, summary reports."
    system_prompt = SYSTEM_PROMPT

    def build_tools(self):
        return [
            make_knowledge_tool(self.slug),
            web_search,
            # *CAPABILITY_TOOLS,  # TODO: add domain-specific capability tools
        ]
