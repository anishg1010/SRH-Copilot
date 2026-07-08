"""Agent registry — the lightweight routing layer.

Agents register themselves with the `@register` decorator. The API and CLI then
discover every agent through `all_agents()` / `get_agent()` without hard-coding
imports. This is what lets you add a new agent by writing one file and importing
it in `copilot.agents` — no changes to the API needed.
"""
from __future__ import annotations

from copilot.core.base_agent import BaseAgent

_REGISTRY: dict[str, BaseAgent] = {}


def register(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Class decorator: instantiate and register an agent by its slug."""
    instance = cls()
    if instance.slug in _REGISTRY:
        raise ValueError(f"Duplicate agent slug: {instance.slug}")
    _REGISTRY[instance.slug] = instance
    return cls


def get_agent(slug: str) -> BaseAgent | None:
    return _REGISTRY.get(slug)


def all_agents() -> list[BaseAgent]:
    return list(_REGISTRY.values())
