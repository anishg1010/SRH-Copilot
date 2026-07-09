"""Step 1 of pre-processing: extract each document to its own JSON file.

This is SEPARATE from the RAG pipeline. It ignores the current 01/02/03 subfolder
structure — the goal is to understand each PDF on its own terms so we can re-categorize.

Output: doc_json/<safe_name>.json  — one file per source document, containing the
full extracted text + block/provenance data + a short preview for quick scanning.

    python -m copilot.preprocess.doc_json ./data/teaching
    python -m copilot.preprocess.doc_json ./data/teaching --out doc_json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from copilot.rag.extractors import SUPPORTED_EXTS, extract


def _safe_name(path: Path, root: Path) -> str:
    """Flatten a path into a filesystem-safe json filename, keeping it unique."""
    rel = path.relative_to(root)
    flat = "__".join(rel.parts)
    flat = re.sub(r"[^A-Za-z0-9._-]+", "_", flat)
    return flat.rsplit(".", 1)[0] + ".json"


def build_doc_json(folder: str, out: str = "doc_json") -> None:
    root = Path(folder)
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    files = [
        p for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
        and ".ipynb_checkpoints" not in p.parts
        and p.name.lower() != "readme.md"
    ]
    if not files:
        raise SystemExit(f"No supported documents under {root}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for f in files:
        try:
            blocks = extract(f)
        except Exception as e:  # noqa: BLE001
            print(f"  ! extract failed: {f.name} ({e})")
            continue
        full_text = "\n".join(b.text for b in blocks).strip()
        if not full_text:
            print(f"  ! no text: {f.name}")
            continue

        record = {
            "source": f.name,
            "original_path": str(f),
            "current_folder": f.parent.name,   # the (inaccurate) existing category
            "num_blocks": len(blocks),
            "num_chars": len(full_text),
            "used_ocr": any(b.meta.get("ocr") for b in blocks),
            "preview": full_text[:1500],       # first ~1500 chars for quick human scan
            "full_text": full_text,
            "blocks": [
                {"text": b.text, "page": b.page, "section": b.section, "meta": b.meta}
                for b in blocks
            ],
        }
        json_name = _safe_name(f, root)
        (out_dir / json_name).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        index.append({
            "json_file": json_name,
            "source": f.name,
            "current_folder": record["current_folder"],
            "num_chars": record["num_chars"],
            "used_ocr": record["used_ocr"],
        })
        print(f"  ✓ {f.name} → {json_name} ({record['num_chars']} chars)")

    (out_dir / "_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✓ {len(index)} document JSON files written to {out_dir}/")
    print(f"  index: {out_dir}/_index.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="doc_json")
    args = ap.parse_args()
    build_doc_json(args.folder, out=args.out)
