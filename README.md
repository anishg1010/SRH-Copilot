# University AI Copilot

A **modular multi-agent platform** for higher education, following the architecture from
the *University AI Copilot* concept: a shared orchestration + knowledge layer with
specialized domain agents on top.

- **Deployment shape:** modular monolith — one FastAPI app, seven self-contained agent
  packages. Microservice-style boundaries without the ops of running seven services;
  any agent can be split out later unchanged.
- **Teaching Innovation Agent is fully built out.** The other six are consistent,
  runnable stubs (RAG + web search wired, domain prompt set) ready to flesh out one at a time.

---

## The seven agents

| Slug | Agent | Status |
|---|---|---|
| `teaching` | Teaching Innovation Agent (CORE Bot) | **Built out** — 4 capability tools |
| `student_services` | Student Service Agent | Stub |
| `feedback` | Feedback Analytics Agent | Stub |
| `career` | Career Support Agent | Stub |
| `hr_compliance` | HR Compliance Agent | Stub |
| `strategic` | Strategic Intelligence Agent | Stub |
| `international` | International Office Agent | Stub |

Every agent — built out or stub — already has retrieval-augmented generation over its own
document collection plus web search. Fleshing one out means adding capability tools.

---

## Architecture

```
                          FastAPI gateway  (api/app.py)
             GET /agents · POST /agents/{slug}/chat
                                  │
                          Agent registry  (core/registry.py)
                                  │  @register auto-discovers agents
   ┌──────────┬──────────┬────────┼────────┬──────────┬──────────┬──────────┐
teaching  student_svc  feedback  career  hr_compliance strategic international
   │          (each subclasses BaseAgent — core/base_agent.py)
   └── BaseAgent provides the LangGraph tool-loop + Postgres checkpointer for all

              Shared services (used by every agent)
   ┌───────────────────────────┬──────────────────────────────┐
   │  RAG  (rag/)              │  Tools (tools/)                │
   │  • collection-scoped      │  • knowledge_search (per agent)│
   │    pgvector store         │  • web_search (shared)         │
   └───────────────┬───────────┴───────────────┬──────────────┘
                   ▼                            ▼
         ┌──────────────────────────────────────────────┐
         │  Postgres + pgvector                          │
         │  • documents  (one table, `collection` column)│
         │  • LangGraph checkpoints (thread = slug:id)   │
         └──────────────────────────────────────────────┘
```

### Why this scales
- **One `BaseAgent`** encodes the tool-calling loop, checkpointer, and message handling
  once. Agents are ~30 lines: a prompt + a tool list.
- **Registry pattern** — add `agents/<slug>/agent.py` with a `@register` class, import it
  in `agents/__init__.py`, and the API/CLI serve it automatically. No gateway edits.
- **One shared Postgres** holds both the vector store (partitioned by `collection`) and
  agent state. Production = a single managed Postgres (RDS / Cloud SQL / Supabase). Split
  a collection into its own table or an agent into its own service only when load demands.

---

## Layout

```
src/copilot/
  core/
    settings.py       # env config
    llm.py            # LLM factory
    state.py          # shared AgentState
    base_agent.py     # BaseAgent: graph loop + checkpointer  ← the reusable core
    registry.py       # @register, get_agent, all_agents
  rag/
    store.py          # pgvector extension + shared documents table
    embeddings.py     # embedding provider
    ingest.py         # ingest one collection
    ingest_all.py     # ingest every collection
    retriever.py      # collection-scoped search
  tools/
    knowledge.py      # make_knowledge_tool(collection) — per-agent RAG
    web.py            # shared web_search
  agents/
    __init__.py       # imports all agents → registers them
    teaching/         # BUILT OUT: agent.py + tools.py (4 capabilities)
    student_services/ …  career/ …  feedback/ …  hr_compliance/ …
    strategic/ …  international/    # stubs: agent.py + tools.py (TODO)
  api/app.py          # FastAPI gateway
  cli.py              # interactive multi-agent CLI
data/<slug>/          # each agent's documents
tests/test_smoke.py
```

---

## Setup

```bash
pip install -e .
docker compose up -d
cp .env.example .env          # add ANTHROPIC_API_KEY, VOYAGE_API_KEY, TAVILY_API_KEY

python -m copilot.rag.store           # create pgvector + documents table
python -m copilot.rag.ingest_all      # ingest all data/<slug> folders (teaching sample included)

python -m copilot.cli teaching        # chat with the teaching agent
# or serve everything:
uvicorn copilot.api.app:app --reload
#   GET  http://localhost:8000/agents
#   POST http://localhost:8000/agents/teaching/chat  {"thread_id":"t1","message":"..."}
```

Drop real documents into `data/<slug>/` and re-run `python -m copilot.rag.ingest <slug> ./data/<slug>`.

---

## Building out a stub agent

1. Open `agents/<slug>/tools.py`, add `@tool` functions (use `retrieve("<slug>", q)` for grounding).
   See `agents/teaching/tools.py` for the pattern.
2. Export them as `CAPABILITY_TOOLS`.
3. In `agents/<slug>/agent.py`, uncomment the import and add `*CAPABILITY_TOOLS` to `build_tools()`.

That's it — no changes to the API, registry, or RAG layer.

---

## Deliberately deferred

The concept's **orchestration layer** (authentication, audit logging, guardrails,
rate-limiting) and **infrastructure** (Kubernetes/Terraform) are not included here — this
is app code only. The `BaseAgent.chat` seam (single entry point per request) and the
per-thread Postgres checkpointer are where auth, logging, and guardrails will slot in later.
```
