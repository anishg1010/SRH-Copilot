"""FastAPI app — one gateway, dynamic routing to every registered agent.

Importing `copilot.agents` registers all agents; the endpoints below then serve
any of them by slug. Adding an agent requires zero changes to this file.

Endpoints:
    GET  /health
    GET  /agents                      → list all agents + their tools
    POST /agents/{slug}/chat          → {"thread_id": "...", "message": "..."}
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import copilot.agents  # noqa: F401  (side effect: registers all agents)
from copilot.core.registry import all_agents, get_agent

app = FastAPI(title="University AI Copilot", version="0.1.0")


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    agent: str
    thread_id: str
    reply: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/agents")
def list_agents() -> list[dict]:
    return [a.info() for a in all_agents()]


@app.post("/agents/{slug}/chat", response_model=ChatResponse)
def chat(slug: str, req: ChatRequest) -> ChatResponse:
    agent = get_agent(slug)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {slug}")
    reply = agent.chat(req.message, thread_id=req.thread_id)
    return ChatResponse(agent=slug, thread_id=req.thread_id, reply=reply)
