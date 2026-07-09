"""Lightweight local agentic layer — tool-calling on the in-venv LLM (transformers).

No LangGraph, no Postgres, no paid API. Uses the same Qwen2.5 model as the RAG layer.
Qwen2.5-Instruct natively supports tool-calling via its chat template: we pass tool
specs, the model emits a tool call, we run the Python function, feed the result back,
and repeat until it produces a final answer.

An agent is: a system prompt + a set of Tool(name, description, json-schema, func).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from copilot.core.config import config


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


@dataclass
class Agent:
    slug: str
    name: str
    description: str
    system_prompt: str
    tools: list[Tool] = field(default_factory=list)

    def _tool_specs(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self.tools
        ]

    def _by_name(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)

    def chat(self, user_message: str, max_steps: int = 5, verbose: bool = True) -> str:
        """Tool-calling loop via Qwen chat template + transformers."""
        import torch
        from copilot.core.llm import _hf_pipeline

        tok, model, device = _hf_pipeline()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        tools = self._tool_specs()

        for _ in range(max_steps):
            text = tok.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
            inputs = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=config.llm_max_tokens,
                    temperature=config.llm_temperature,
                    do_sample=config.llm_temperature > 0,
                    pad_token_id=tok.eos_token_id,
                )
            reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            calls = _parse_tool_calls(reply)
            if not calls:
                return reply

            messages.append({"role": "assistant", "content": reply})
            for name, args in calls:
                if verbose:
                    print(f"  · tool: {name}({args})")
                tool = self._by_name(name)
                result = tool.func(**args) if tool else f"[unknown tool: {name}]"
                messages.append({"role": "tool", "name": name, "content": str(result)})

        # final pass without tools
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=config.llm_max_tokens,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Extract tool calls from Qwen output.

    Qwen emits calls inside <tool_call>...</tool_call> as JSON {"name":..,"arguments":..}.
    """
    calls = []
    for m in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            obj = json.loads(m)
            name = obj.get("name")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                calls.append((name, args))
        except Exception:
            continue
    return calls
