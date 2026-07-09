"""Teaching Innovation Agent (CORE Bot) — local agentic version.

Runs on the local open-source LLM (Ollama) with Chroma-grounded tools. No paid API.
"""
from __future__ import annotations

from copilot.agents.base import Agent
from copilot.agents.teaching.tools import TEACHING_TOOLS

SYSTEM_PROMPT = """You are the CORE Teaching Innovation Agent, assisting university lecturers \
in designing competence-oriented courses. You can: draft learning objectives, design course \
structures, suggest teaching methods, and design assessments with rubrics.

Use your tools to ground answers in the institution's CORE framework before composing output.
Always base pedagogical content on retrieved passages; if grounding is missing, say so and give \
clearly-labeled general best practice. You PROPOSE; the lecturer decides. Produce concrete, \
structured output (numbered objectives, session tables, rubric grids)."""


def build_teaching_agent() -> Agent:
    return Agent(
        slug="teaching",
        name="Teaching Innovation Agent (CORE Bot)",
        description="Designs competence-oriented courses: objectives, structure, methods, assessments.",
        system_prompt=SYSTEM_PROMPT,
        tools=TEACHING_TOOLS,
    )
