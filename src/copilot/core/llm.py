"""Local open-source LLM. Default provider: in-process transformers (GPU), no Ollama.

Providers (LLM_PROVIDER in .env):
  "transformers" (DEFAULT) — loads config.hf_model (Qwen2.5-7B-Instruct) in-venv on GPU.
  "ollama"                 — talks to a local Ollama server (if you install one).
  "anthropic"              — Claude API (later, when tokens provided).

The transformers model is loaded lazily and cached, so import is cheap and the model
is only pulled/loaded the first time you actually generate.
"""
from __future__ import annotations

from functools import lru_cache

from copilot.core.config import config


class LLMUnavailable(RuntimeError):
    pass


# ── transformers (default) ─────────────────────────────────────
@lru_cache(maxsize=1)
def _hf_pipeline():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config.llm_device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(config.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.hf_model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    return tok, model, device


def _generate_transformers(prompt: str, system: str | None) -> str:
    import torch
    tok, model, device = _hf_pipeline()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
            do_sample=config.llm_temperature > 0,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


# ── ollama ─────────────────────────────────────────────────────
def _generate_ollama(prompt: str, system: str | None) -> str:
    import httpx
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = httpx.post(
            f"{config.ollama_base_url}/api/chat",
            json={"model": config.ollama_model, "messages": messages, "stream": False,
                  "options": {"temperature": config.llm_temperature,
                              "num_predict": config.llm_max_tokens}},
            timeout=180.0,
        )
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise LLMUnavailable(f"Cannot reach Ollama: {e}") from e
    return resp.json()["message"]["content"].strip()


# ── anthropic (later) ──────────────────────────────────────────
def _generate_anthropic(prompt: str, system: str | None) -> str:
    import anthropic
    if not config.anthropic_api_key:
        raise LLMUnavailable("llm_provider=anthropic but ANTHROPIC_API_KEY not set.")
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    msg = client.messages.create(
        model=config.anthropic_model, max_tokens=config.llm_max_tokens,
        temperature=config.llm_temperature, system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def generate(prompt: str, system: str | None = None) -> str:
    if config.llm_provider == "ollama":
        return _generate_ollama(prompt, system)
    if config.llm_provider == "anthropic":
        return _generate_anthropic(prompt, system)
    return _generate_transformers(prompt, system)
