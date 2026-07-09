# University SRH Copilot

Intelligent support for teaching, grounded in SRH's own knowledge base. The first
module is the **Teaching Innovation Agent (CORE Bot)**, which helps lecturers design
competence-oriented courses: learning objectives, course & module design, teaching
methods, assessments & rubrics, and responsible AI integration.

Everything runs **locally on open-source models** — no paid API, no data leaves the machine.

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │   University SRH Copilot (Streamlit)   │  ← product frontend
                    └───────────────────┬──────────────────┘
                                        │
                    ┌───────────────────▼──────────────────┐
                    │   Teaching Innovation Agent (CORE Bot) │  ← tool-calling agent
                    │   objectives · course · methods ·      │
                    │   assessment · responsible AI          │
                    └───────────────────┬──────────────────┘
                                        │ retrieval-augmented generation
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                                ▼                               ▼
  embeddings (e5)              ChromaDB (vectors)              LLM (Qwen2.5-7B)
  multilingual DE/EN            local, persistent               local, GPU
```

## Project structure

```
university_srh_copilot/
├── app/
│   └── streamlit_app.py         Product frontend (customer-facing)
├── src/copilot/
│   ├── core/                    config + LLM provider
│   ├── rag/                     extraction, chunking, embeddings, store,
│   │                            retrieval, generation
│   ├── preprocess/              document understanding & categorization
│   └── agents/
│       ├── base.py              tool-calling agent runtime
│       └── teaching/            Teaching Innovation Agent + its tools
├── notebooks/                   technical visualizations (team / manager)
│   ├── 01_rag_pipeline.ipynb
│   ├── 02_document_categorization.ipynb
│   └── 03_teaching_agent.ipynb
├── tests/                       automated tests
├── data/teaching/              raw source documents (fallback)
├── data_categorized/           documents organized by teaching function
└── docs/
```

## Models (local, open-source)

| Role | Model | Notes |
|------|-------|-------|
| Embeddings | `intfloat/multilingual-e5-large` | Strong German + English, 1024-dim |
| LLM | `Qwen/Qwen2.5-7B-Instruct` | In-process via transformers, GPU |
| Vector store | ChromaDB | Local, persistent |

All configurable in `src/copilot/core/config.py` (override via `.env`).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebooks]"
# OCR for scanned PDFs (optional):
sudo apt-get install -y tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng poppler-utils
```

## Pipeline

```bash
# 1. (optional) understand & re-categorize raw documents by teaching function
python -m copilot.preprocess.doc_json ./data/teaching --out doc_json
python -m copilot.preprocess.categorize --doc-json doc_json
#    edit categorization_mapping.csv, then:
python -m copilot.preprocess.apply_mapping categorization_mapping.csv \
    --source-root data/teaching --dest data_categorized

# 2. build the knowledge base
python -m copilot.rag.ingest teaching ./data_categorized --reset

# 3a. run the product frontend
streamlit run app/streamlit_app.py

# 3b. or use the agent from the terminal
python -m copilot.agents.cli

# 3c. or a direct RAG answer
python -m copilot.rag.generate teaching "How does the CORE principle structure modules?"
```

## Audiences

- **Customers / lecturers** → the Streamlit app (`app/streamlit_app.py`).
- **Technical team / manager** → the notebooks (pipeline, clustering, agent internals).
- **Developers** → `src/` + `tests/`.
