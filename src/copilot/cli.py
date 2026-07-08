"""Interactive CLI — pick any registered agent and chat.

    python -m copilot.cli                 # lists agents, prompts for one
    python -m copilot.cli teaching        # jump straight to an agent
"""
from __future__ import annotations

import sys
import uuid

import copilot.agents  # noqa: F401  (registers all agents)
from copilot.core.registry import all_agents, get_agent


def main() -> None:
    agents = all_agents()
    slug = sys.argv[1] if len(sys.argv) > 1 else None

    if slug is None:
        print("Available agents:")
        for a in agents:
            print(f"  {a.slug:18s} — {a.name}")
        slug = input("\nPick an agent slug > ").strip()

    agent = get_agent(slug)
    if agent is None:
        sys.exit(f"Unknown agent: {slug}")

    thread_id = str(uuid.uuid4())
    print(f"\n{agent.name} — type 'exit' to quit. (thread: {thread_id})\n")
    while True:
        try:
            user = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        print("\nAgent >", agent.chat(user, thread_id=thread_id), "\n")


if __name__ == "__main__":
    main()
