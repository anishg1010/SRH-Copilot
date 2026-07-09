"""University SRH Copilot — product frontend (Streamlit).

The customer-facing interface for the Teaching Innovation Agent (CORE Bot). Lecturers
ask questions; the agent grounds answers in the institution's teaching knowledge base
and cites its sources.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ─── Page config & branding ────────────────────────────────────
st.set_page_config(
    page_title="University SRH Copilot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

SRH_ORANGE = "#EA5B0C"

st.markdown(f"""
<style>
    .stApp {{ background-color: #0f1116; }}
    h1, h2, h3 {{ color: #ffffff; }}
    .srh-header {{
        display: flex; align-items: center; gap: 0.75rem;
        padding: 0.5rem 0 1rem 0; border-bottom: 2px solid {SRH_ORANGE};
        margin-bottom: 1.5rem;
    }}
    .srh-logo {{
        font-weight: 800; font-size: 1.6rem; color: {SRH_ORANGE};
        letter-spacing: -1px;
    }}
    .srh-title {{ font-size: 1.4rem; font-weight: 700; color: #fff; }}
    .srh-sub {{ color: #9aa0aa; font-size: 0.9rem; }}
    .source-card {{
        background: #1a1d24; border-left: 3px solid {SRH_ORANGE};
        padding: 0.6rem 0.9rem; margin: 0.4rem 0; border-radius: 4px;
        font-size: 0.85rem; color: #cfd3da;
    }}
    .source-meta {{ color: {SRH_ORANGE}; font-size: 0.78rem; font-weight: 600; }}
    .stChatMessage {{ background: #161922; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="srh-header">
    <span class="srh-logo">srh</span>
    <div>
        <div class="srh-title">University SRH Copilot</div>
        <div class="srh-sub">Teaching Innovation Agent · CORE Bot</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.caption(
        "Intelligent support for competence-oriented teaching. The CORE Bot helps "
        "lecturers draft learning objectives, design courses and modules, suggest "
        "teaching methods, and build assessments — grounded in SRH's teaching framework."
    )
    st.markdown("### Try asking")
    examples = [
        "Draft 5 learning objectives for a Bachelor module on Data Ethics",
        "Suggest active-learning methods for critical AI literacy",
        "Design an assessment with a rubric for evaluating AI-generated text",
        "How does the CORE principle structure modules?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.pending = ex

    st.markdown("---")
    show_sources = st.toggle("Show retrieved sources", value=True)
    mode = st.radio("Mode", ["Agent (uses tools)", "Direct RAG answer"], index=0)


# ─── Lazy-load the agent / RAG (cached) ────────────────────────
@st.cache_resource(show_spinner="Loading the CORE Bot… (first load pulls the model)")
def _load_agent():
    from copilot.agents.teaching.agent import build_teaching_agent
    return build_teaching_agent()


def _rag_answer(question: str):
    from copilot.rag.generate import answer
    return answer("teaching", question)


# ─── Chat state ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            for s in msg["sources"]:
                st.markdown(
                    f'<div class="source-card"><span class="source-meta">'
                    f'{s["source"]} · {s.get("topic","")} · relevance {s["score"]:.2f}'
                    f'</span><br>{s["preview"]}</div>',
                    unsafe_allow_html=True,
                )

# input
user_input = st.chat_input("Ask the CORE Bot about teaching, courses, or assessment…")
if st.session_state.pending and not user_input:
    user_input = st.session_state.pending
    st.session_state.pending = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking…"):
            sources = []
            try:
                if mode.startswith("Direct"):
                    result = _rag_answer(user_input)
                    reply = result["answer"]
                    sources = [
                        {"source": p.source, "topic": p.topic or "",
                         "score": p.score, "preview": p.content[:220] + "…"}
                        for p in result["passages"]
                    ]
                else:
                    agent = _load_agent()
                    reply = agent.chat(user_input, verbose=False)
            except Exception as e:  # noqa: BLE001
                reply = f"⚠️ Something went wrong: {e}"
        st.markdown(reply)
        if sources and show_sources:
            for s in sources:
                st.markdown(
                    f'<div class="source-card"><span class="source-meta">'
                    f'{s["source"]} · {s["topic"]} · relevance {s["score"]:.2f}'
                    f'</span><br>{s["preview"]}</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
