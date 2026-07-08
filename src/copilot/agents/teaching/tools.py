"""Teaching Innovation Agent (CORE Bot) — the four capability tools.

Each tool retrieves CORE grounding from the 'teaching' collection, then returns a
focused task brief the agent's LLM uses to compose the final artifact.
"""
from __future__ import annotations

from langchain_core.tools import tool

from copilot.rag.retriever import format_passages, retrieve

COLLECTION = "teaching"

# Topic tags = the folder names under data/teaching/structured/. Scoping retrieval to
# the relevant topics is the single biggest retrieval-quality win on a broad corpus:
# it keeps, e.g., the assessment tool grounded in CORE/didactics material rather than
# the whole AI-ethics reading list. Adjust these to match YOUR actual folder names.
TOPIC_CORE = ["02_CORE_Principle_and_Modular_Design"]
TOPIC_DIDACTICS = ["03_Didactics_and_Instructional_Design"]
TOPIC_AI = ["01_AI_in_Higher_Education_and_Writing"]


def _ground(query: str, topics: list[str] | None = None) -> str:
    # min_score drops weak matches; tune the threshold in settings/experimentation.
    return format_passages(retrieve(COLLECTION, query, topics=topics, min_score=0.20))


@tool
def draft_learning_objectives(topic: str, level: str = "Bachelor", count: int = 5) -> str:
    """Draft competence-oriented, measurable learning objectives for a course/module.

    Args:
        topic: Subject of the course or module.
        level: Study level ("Bachelor", "Master", or a specific semester).
        count: How many objectives to draft.
    """
    grounding = _ground(f"competence-oriented learning objectives {topic} {level}",
                        topics=TOPIC_CORE + TOPIC_DIDACTICS)
    return (
        f'TASK: Draft {count} competence-oriented learning objectives for "{topic}" '
        f"at {level} level.\n\n"
        "REQUIREMENTS:\n"
        "- Start each with an observable, measurable action verb (Bloom-aligned).\n"
        "- Span cognitive levels (remember → create), not all low-level.\n"
        "- Phrase as competences the student demonstrates, using CORE language where present.\n"
        "- Avoid vague verbs (understand, know, learn about).\n\n"
        f"CORE FRAMEWORK GROUNDING:\n{grounding}"
    )


@tool
def design_course_structure(topic: str, weeks: int = 12, level: str = "Bachelor") -> str:
    """Design a session-by-session course/module structure.

    Args:
        topic: Course/module subject.
        weeks: Number of sessions/weeks.
        level: Study level.
    """
    grounding = _ground(f"module structure course design template {topic} {level}",
                        topics=TOPIC_CORE + TOPIC_DIDACTICS)
    return (
        f'TASK: Design a {weeks}-session structure for "{topic}" ({level}).\n\n'
        "REQUIREMENTS:\n"
        "- Per session: title, objective(s) advanced, key topics, in-class activity, prep/homework.\n"
        "- Sequence foundational → applied; show constructive alignment to assessment.\n"
        "- Follow the institution's module template where the grounding specifies it.\n\n"
        f"CORE FRAMEWORK GROUNDING:\n{grounding}"
    )


@tool
def suggest_teaching_methods(objective: str, constraints: str = "") -> str:
    """Suggest active-learning methods matched to a learning objective.

    Args:
        objective: The learning objective/competence to develop.
        constraints: Optional context (class size, online/hybrid, time, resources).
    """
    grounding = _ground(f"active learning teaching methods {objective}",
                        topics=TOPIC_DIDACTICS + TOPIC_AI)
    return (
        f'TASK: Recommend teaching methods for the objective:\n"{objective}"\n'
        f"CONSTRAINTS: {constraints or 'none specified'}\n\n"
        "REQUIREMENTS:\n"
        "- Propose 3-5 concrete methods; for each explain fit to the cognitive level and how to run it.\n"
        "- Include one option that responsibly integrates AI as a learning aid, per CORE guidance.\n"
        "- Note assessment implications of each.\n\n"
        f"CORE FRAMEWORK GROUNDING:\n{grounding}"
    )


@tool
def design_assessment(objective: str, assessment_type: str = "auto") -> str:
    """Design an assessment with an analytic rubric, aligned to a learning objective.

    Args:
        objective: The learning objective to measure.
        assessment_type: "exam", "project", "presentation", "portfolio", or "auto".
    """
    grounding = _ground(f"assessment rubric criteria {assessment_type} {objective}",
                        topics=TOPIC_CORE + TOPIC_DIDACTICS)
    return (
        f'TASK: Design an assessment (type: {assessment_type}) measuring:\n"{objective}"\n\n'
        "REQUIREMENTS:\n"
        "- Justify format alignment to the objective's competence level.\n"
        "- Provide the task/prompt students receive.\n"
        "- Provide an analytic rubric: 3-5 criteria × levels (Insufficient/Sufficient/Good/Excellent).\n"
        "- Ensure criteria are observable and map to the objective.\n"
        "- Follow institutional grading policy where the grounding specifies it.\n\n"
        f"CORE FRAMEWORK GROUNDING:\n{grounding}"
    )


TEACHING_CAPABILITY_TOOLS = [
    draft_learning_objectives,
    design_course_structure,
    suggest_teaching_methods,
    design_assessment,
]
