"""Capability tools for the Student Service Agent — TODO.

Follow the pattern in agents/teaching/tools.py: each tool retrieves grounding from the
'student_services' collection via `retrieve("student_services", query)`, then returns a focused task brief
(or a direct result). Export them as a list and import it in agent.py.
"""
from __future__ import annotations

from copilot.rag.retriever import format_passages, retrieve

COLLECTION = "student_services"


def _ground(query: str) -> str:
    return format_passages(retrieve(COLLECTION, query))


# TODO: define @tool functions here, e.g. cluster_feedback, review_cv, monitor_competitors ...

CAPABILITY_TOOLS: list = []
