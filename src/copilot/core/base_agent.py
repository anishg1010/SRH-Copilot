"""BaseAgent — the reusable pattern every domain agent inherits.

Each agent is defined by three things:
  1. a `slug`   — stable id used in the API path and registry
  2. a `name` + `description` — human-facing metadata
  3. a `system_prompt` and a list of `tools`

Everything else (the LangGraph tool-calling loop, the Postgres checkpointer,
message handling) is provided here so individual agents stay tiny and consistent.

Subclass, set the class attributes / implement `build_tools()`, and register it:

    @register
    class MyAgent(BaseAgent):
        slug = "my_agent"
        name = "My Agent"
        description = "..."
        system_prompt = "..."
        def build_tools(self):
            return [tool_a, tool_b]
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from copilot.core.llm import make_llm
from copilot.core.settings import settings
from copilot.core.state import AgentState


class BaseAgent:
    slug: str = ""
    name: str = ""
    description: str = ""
    system_prompt: str = "You are a helpful university assistant."
    model: str | None = None  # override per-agent if desired

    def __init__(self) -> None:
        if not self.slug:
            raise ValueError(f"{type(self).__name__} must define a `slug`.")
        self._tools: list[BaseTool] = self.build_tools()

    # --- to be overridden by subclasses ---------------------------------
    def build_tools(self) -> list[BaseTool]:
        """Return the tools this agent can call. Override in subclasses."""
        return []

    # --- shared graph machinery -----------------------------------------
    def _agent_node(self, state: AgentState) -> dict:
        llm = make_llm(self.model).bind_tools(self._tools) if self._tools else make_llm(self.model)
        messages = [SystemMessage(content=self.system_prompt), *state["messages"]]
        return {"messages": [llm.invoke(messages)]}

    def _skeleton(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent_node)
        g.add_edge(START, "agent")
        if self._tools:
            g.add_node("tools", ToolNode(self._tools))
            g.add_conditional_edges("agent", tools_condition)
            g.add_edge("tools", "agent")
        return g

    @contextmanager
    def compiled(self) -> Iterator:
        """Yield a compiled graph with a live Postgres checkpointer.

        Checkpoints are namespaced per agent via the thread_id convention
        `{slug}:{thread_id}` set in `chat()`, so all agents can share one DB.
        """
        with PostgresSaver.from_conn_string(settings.database_url) as cp:
            cp.setup()
            yield self._skeleton().compile(checkpointer=cp)

    def chat(self, message: str, thread_id: str) -> str:
        """One-shot invoke that persists to the shared checkpointer."""
        config = {"configurable": {"thread_id": f"{self.slug}:{thread_id}"}}
        with self.compiled() as graph:
            result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
        return result["messages"][-1].content

    def info(self) -> dict:
        return {
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "tools": [t.name for t in self._tools],
        }
