"""Offline smoke tests: registry wiring, agent structure, chunking — no DB/API calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_all_seven_agents_registered():
    import copilot.agents  # noqa: F401
    from copilot.core.registry import all_agents
    slugs = {a.slug for a in all_agents()}
    expected = {
        "teaching", "student_services", "feedback", "career",
        "hr_compliance", "strategic", "international",
    }
    assert slugs == expected, f"missing/extra: {slugs ^ expected}"


def test_every_agent_has_knowledge_and_web_tools():
    import copilot.agents  # noqa: F401
    from copilot.core.registry import all_agents
    for a in all_agents():
        names = {t.name for t in a._tools}
        assert "knowledge_search" in names, f"{a.slug} missing knowledge_search"
        assert "web_search" in names, f"{a.slug} missing web_search"


def test_teaching_agent_fully_built():
    import copilot.agents  # noqa: F401
    from copilot.core.registry import get_agent
    names = {t.name for t in get_agent("teaching")._tools}
    for t in ("draft_learning_objectives", "design_course_structure",
              "suggest_teaching_methods", "design_assessment"):
        assert t in names, f"teaching missing {t}"


def test_chunking():
    from copilot.rag.ingest import _chunk
    chunks = _chunk("word " * 2000, size=500, overlap=60)
    assert len(chunks) > 1 and all(chunks)


def test_vec_literal():
    from copilot.rag.embeddings import vec_literal
    assert vec_literal([0.1, 0.2]) == "[0.100000,0.200000]"


if __name__ == "__main__":
    for fn in [
        test_all_seven_agents_registered,
        test_every_agent_has_knowledge_and_web_tools,
        test_teaching_agent_fully_built,
        test_chunking,
        test_vec_literal,
    ]:
        fn()
        print(f"✓ {fn.__name__}")
    print("\n✓ all smoke tests passed")


def test_extractors_dispatch_and_supported_exts():
    from copilot.rag.extractors import SUPPORTED_EXTS
    for e in (".pdf", ".docx", ".pptx", ".txt", ".md", ".jpg", ".png"):
        assert e in SUPPORTED_EXTS


def test_structure_aware_chunking_respects_sections():
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
        assert not ("alpha" in c.text and "beta" in c.text)  # sections never merge
