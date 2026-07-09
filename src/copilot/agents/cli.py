"""Run the teaching agent interactively (terminal).

    python -m copilot.agents.cli
"""
from __future__ import annotations

from copilot.agents.teaching.agent import build_teaching_agent


def main() -> None:
    agent = build_teaching_agent()
    print(f"{agent.name} — type 'exit' to quit.\n")
    while True:
        try:
            user = input("Lecturer > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        try:
            print(f"\nAgent > {agent.chat(user)}\n")
        except Exception as e:  # noqa: BLE001
            print(f"\n[error] {e}\n")


if __name__ == "__main__":
    main()
