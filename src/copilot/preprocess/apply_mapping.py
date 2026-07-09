"""Step 3 of pre-processing: apply the (edited) mapping CSV to build the new,
function-based folder structure.

Reads categorization_mapping.csv (after you've corrected any 'proposed_category'
values) and COPIES each source document into dest/<category>/<file>. Copies, not
moves — your originals stay untouched.

    python -m copilot.preprocess.apply_mapping categorization_mapping.csv \
        --source-root data/teaching --dest data_recategorized
    # add --move to move instead of copy, --dry-run to preview
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def apply_mapping(csv_path: str, source_root: str, dest: str,
                  move: bool = False, dry_run: bool = False) -> None:
    src_root = Path(source_root)
    dest_root = Path(dest)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    # build a lookup of source filename → its actual path on disk (recursive)
    by_name: dict[str, Path] = {}
    for p in src_root.rglob("*"):
        if p.is_file():
            by_name.setdefault(p.name, p)

    moved, missing = 0, []
    per_cat: dict[str, int] = {}
    for r in rows:
        name = r["source"]
        category = (r.get("proposed_category") or "uncategorized").strip()
        src = by_name.get(name)
        if not src:
            missing.append(name)
            continue
        target_dir = dest_root / category
        target = target_dir / name
        per_cat[category] = per_cat.get(category, 0) + 1
        if dry_run:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(target))
        else:
            shutil.copy2(str(src), str(target))
        moved += 1

    print(("— DRY RUN —\n" if dry_run else "") + f"{'Would place' if dry_run else 'Placed'} files by category:")
    for cat, n in sorted(per_cat.items()):
        print(f"    {cat:22s} {n}")
    if not dry_run:
        print(f"\n✓ {moved} files {'moved' if move else 'copied'} into {dest_root}/")
    if missing:
        print(f"\n⚠ {len(missing)} source file(s) not found under {src_root}:")
        for m in missing[:20]:
            print(f"    {m}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--source-root", default="data/teaching")
    ap.add_argument("--dest", default="data_recategorized")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    apply_mapping(args.csv_path, args.source_root, args.dest,
                  move=args.move, dry_run=args.dry_run)
