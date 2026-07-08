"""Importing this package registers every agent via the @register decorator.

Add a new agent by creating agents/<slug>/agent.py with a @register class, then
importing it here. The API and CLI pick it up automatically — no other changes.
"""
from copilot.agents.teaching.agent import TeachingAgent
from copilot.agents.student_services.agent import StudentServicesAgent
from copilot.agents.feedback.agent import FeedbackAnalyticsAgent
from copilot.agents.career.agent import CareerSupportAgent
from copilot.agents.hr_compliance.agent import HRComplianceAgent
from copilot.agents.strategic.agent import StrategicIntelligenceAgent
from copilot.agents.international.agent import InternationalOfficeAgent

__all__ = [
    "TeachingAgent",
    "StudentServicesAgent",
    "FeedbackAnalyticsAgent",
    "CareerSupportAgent",
    "HRComplianceAgent",
    "StrategicIntelligenceAgent",
    "InternationalOfficeAgent",
]
