"""Teaching Innovation Agent (CORE Bot) — capability tools.

Each tool retrieves grounding from the 'teaching' collection, scoped to the relevant
function-based topic categories (from the document pre-processing step), then returns a
concise brief the LLM uses to compose the final artifact. Tools are plain functions
wrapped as Tool with a JSON schema so the local model can call them.

Topic categories match the Teaching Agent's functions:
  learning_objectives · course_module_design · teaching_methods
  assessment_rubrics · responsible_ai · general_core
"""
from __future__ import annotations

from copilot.agents.base import Tool
from copilot.rag.retriever import format_passages, retrieve

COLLECTION = "teaching"

# Function-based topic tags (= folders under data_categorized/, from preprocessing).
T_OBJECTIVES = ["learning_objectives"]
T_COURSE = ["course_module_design", "general_core"]
T_METHODS = ["teaching_methods"]
T_ASSESSMENT = ["assessment_rubrics"]
T_AI = ["responsible_ai"]


def _ground(query: str, topics=None) -> str:
    return format_passages(retrieve(COLLECTION, query, topics=topics, min_score=0.15))


def knowledge_search(query: str) -> str:
    return _ground(query)


def draft_learning_objectives(topic: str, level: str = "Bachelor", count: int = 5) -> str:
    g = _ground(f"competence-oriented learning objectives {topic} {level}",
                topics=T_OBJECTIVES + T_COURSE)
    return (f"Draft {count} competence-oriented, measurable learning objectives for "
            f"'{topic}' ({level}). Use observable action verbs, span cognitive levels, "
            f"avoid vague verbs. Ground in:\n{g}")


def design_course_structure(topic: str, weeks: int = 12, level: str = "Bachelor") -> str:
    g = _ground(f"module structure course design {topic} {level}", topics=T_COURSE)
    return (f"Design a {weeks}-session structure for '{topic}' ({level}): per session give "
            f"title, objectives, topics, activity, prep. Show constructive alignment. "
            f"Ground in:\n{g}")


def suggest_teaching_methods(objective: str, constraints: str = "") -> str:
    g = _ground(f"active learning teaching methods {objective}", topics=T_METHODS + T_AI)
    return (f"Recommend 3-5 active-learning methods for: '{objective}'. Constraints: "
            f"{constraints or 'none'}. Explain fit + how to run each. Ground in:\n{g}")


def design_assessment(objective: str, assessment_type: str = "auto") -> str:
    g = _ground(f"assessment rubric criteria {assessment_type} {objective}",
                topics=T_ASSESSMENT + T_COURSE)
    return (f"Design an assessment (type: {assessment_type}) for: '{objective}'. Include the "
            f"task and an analytic rubric (3-5 criteria x levels). Ground in:\n{g}")


TEACHING_TOOLS = [
    Tool("knowledge_search", "Search the teaching knowledge base for grounding passages.",
         {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
         knowledge_search),
    Tool("draft_learning_objectives", "Draft competence-oriented learning objectives.",
         {"type": "object", "properties": {
             "topic": {"type": "string"}, "level": {"type": "string"}, "count": {"type": "integer"}},
          "required": ["topic"]}, draft_learning_objectives),
    Tool("design_course_structure", "Design a session-by-session course/module structure.",
         {"type": "object", "properties": {
             "topic": {"type": "string"}, "weeks": {"type": "integer"}, "level": {"type": "string"}},
          "required": ["topic"]}, design_course_structure),
    Tool("suggest_teaching_methods", "Suggest active-learning methods for an objective.",
         {"type": "object", "properties": {
             "objective": {"type": "string"}, "constraints": {"type": "string"}},
          "required": ["objective"]}, suggest_teaching_methods),
    Tool("design_assessment", "Design an assessment with a rubric for an objective.",
         {"type": "object", "properties": {
             "objective": {"type": "string"}, "assessment_type": {"type": "string"}},
          "required": ["objective"]}, design_assessment),
]
