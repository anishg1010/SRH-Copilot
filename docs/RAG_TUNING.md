# RAG Tuning Guide (Teaching Agent)

This corpus is a broad, bilingual (DE/EN) research library — books, papers, frameworks,
guidelines — organized into topic subfolders. That shape drives the choices below.

## Pipeline overview

```
data/teaching/structured/<NN_Topic>/<file>   ← your files (folder = topic tag)
        │  extractors.py  (pdf/docx/pptx/txt/md/img + OCR fallback)
        ▼  → Blocks (text + page + section)
   chunking.py  (section-coherent, page-tracked, size-limited)
        ▼  → Chunks
   ingest.py    (embed + store with metadata {topic, page_start/end, section, source})
        ▼
   Postgres + pgvector  (documents table; GIN index on metadata for topic filters)
        ▲
   retriever.py  (collection + optional topic filter + min_score)
```

## Step-by-step: get your docs in

```bash
# 1. Dry run FIRST — extracts, chunks, and reports, but writes nothing.
#    Tells you chunk counts per topic, which files needed OCR, and which yielded no text.
python -m copilot.rag.ingest teaching ./data/teaching --dry-run

# 2. If the report looks sane, ingest for real (fresh each time with --reset):
python -m copilot.rag.ingest teaching ./data/teaching --reset

# 3. Inspect what landed:
python -m copilot.rag.inspect topics teaching

# 4. Test retrieval interactively (this is where you tune):
python -m copilot.rag.inspect query teaching "How does the CORE principle structure modules?"
python -m copilot.rag.inspect query teaching "assessment rubric criteria" --topics 02_CORE_Principle_and_Modular_Design --k 8
python -m copilot.rag.inspect query teaching "generative AI writing policy" --min-score 0.25
```

## The knobs you tune (in `.env` / settings)

| Knob | Default | Effect | For this corpus |
|---|---|---|---|
| `CHUNK_TOKENS` | 500 | chunk size | Books/papers → try 500–800. Larger = more context per hit, fewer chunks. |
| `CHUNK_OVERLAP` | 60 | token overlap | 10–15% of chunk size. Prevents losing ideas split across a boundary. |
| `RETRIEVE_K` | 6 | passages returned | Broad corpus → 6–10. Higher recall, more noise. |
| `min_score` (per call) | none | similarity floor | 0.20–0.30 drops junk. Set per query in `inspect`, and in tools via `_ground`. |
| `topics` (per call) | none | restrict to folders | The biggest lever — scope tools to relevant topics. |

Change `.env`, then **re-ingest** if you touched `CHUNK_*` (they affect stored chunks),
or just re-query if you only touched `RETRIEVE_K` / `min_score` / `topics` (query-time).

## OCR requirements

OCR (scanned PDFs + the CORE Standards JPGs) needs the Tesseract **system binary** with
German + English language data. In your JupyterLab terminal:

```bash
# if you have sudo / a package manager:
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng poppler-utils
# poppler-utils provides pdftoppm, needed by pdf2image for scanned-PDF OCR.
```

If you can't install system packages in the lab, OCR silently falls back to empty — the
`--dry-run` report lists every file that yielded no text, so you'll see exactly which
scanned files need OCR. You can OCR those elsewhere and drop in text versions.

Python deps (in `pyproject.toml`): `python-docx`, `python-pptx`, `pdf2image`,
`pytesseract`, `Pillow`. Install with `pip install -e .`.

## Topic scoping in the tools

`agents/teaching/tools.py` defines `TOPIC_CORE`, `TOPIC_DIDACTICS`, `TOPIC_AI` and scopes
each capability tool to the relevant ones. **Edit these lists to match your real folder
names** (run `inspect topics teaching` to see the exact strings). Example: the assessment
tool grounds in CORE + didactics, not the AI-ethics readings.

## A minimal eval loop (recommended before trusting output)

Pick ~10 questions a lecturer would ask, note which source *should* answer each, then run
them through `inspect query` and check the right source appears in the top-k. When you
change a knob, re-run the same 10 and see if hit-rate improves. That turns tuning from
guesswork into measurement. Ask and I'll scaffold a small `eval.py` that scores this
automatically.
