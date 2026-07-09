"""Offline smoke tests — no GPU, no models, no network.

    python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_config_loads():
    from copilot.core.config import config
    assert config.embed_provider in ("local", "chroma", "voyage")
    assert config.llm_provider in ("transformers", "ollama", "anthropic")
    assert config.chunk_tokens > 0


def test_teaching_agent_builds():
    from copilot.agents.teaching.agent import build_teaching_agent
    agent = build_teaching_agent()
    names = {t.name for t in agent.tools}
    for t in ("knowledge_search", "draft_learning_objectives", "design_course_structure",
              "suggest_teaching_methods", "design_assessment"):
        assert t in names, f"missing tool {t}"


def test_extractors_supported_exts():
    from copilot.rag.extractors import SUPPORTED_EXTS
    for e in (".pdf", ".docx", ".pptx", ".txt", ".md", ".jpg", ".png"):
        assert e in SUPPORTED_EXTS


def test_structure_aware_chunking():
    from copilot.rag.extractors import Block
    from copilot.rag.chunking import chunk_blocks
    blocks = [
        Block(text="Section A", page=1, section="Section A", meta={"heading": True}),
        Block(text="alpha " * 40, page=1, section="Section A"),
        Block(text="Section B", page=2, section="Section B", meta={"heading": True}),
        Block(text="beta " * 40, page=2, section="Section B"),
    ]
    chunks = chunk_blocks(blocks, size=100, overlap=15)
    assert chunks
    for c in chunks:
        assert not ("alpha" in c.text and "beta" in c.text)


if __name__ == "__main__":
    tests = [test_config_loads, test_teaching_agent_builds,
             test_extractors_supported_exts, test_structure_aware_chunking]
    for fn in tests:
        fn()
        print(f"✓ {fn.__name__}")
    print(f"\n✓ all {len(tests)} smoke tests passed")
