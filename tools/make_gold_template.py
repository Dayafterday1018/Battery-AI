"""Prepare the OCR gold set: 40 normal-battery photos whose brand a person has checked.

Picks photos from data/raw/normal (seed 42), copies them to data/ocr_gold/ and writes
data/ocr_gold.csv. The brand column is pre-filled from data/brand_labels.csv when the class
data came with brand folders; otherwise it is left empty for the team to fill in.
Also saves numbered contact sheets (reports/gold_sheet_*.png) to label from.

Run: python tools/make_gold_template.py [--n 40]
Then open data/ocr_gold.csv, fill every empty brand (use the exact names in data/brands.csv,
or delete the row if no brand is readable), and have a second teammate check it.
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import OCR_GOLD_CSV, RAW, REPORTS, SEED  # noqa: E402
from imaging import list_images, open_rgb  # noqa: E402


def sheet(paths, out, start):
    cols, thumb = 5, 220
    rows = (len(paths) + cols - 1) // cols
    img = Image.new("RGB", (cols * thumb, rows * (thumb + 22)), "white")
    d = ImageDraw.Draw(img)
    for k, p in enumerate(paths):
        t = open_rgb(p)
        t.thumbnail((thumb, thumb))
        x, y = (k % cols) * thumb, (k // cols) * (thumb + 22)
        img.paste(t, (x, y))
        d.text((x + 4, y + thumb + 4), f"#{start + k}  {p.name[:26]}", fill="black")
    img.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--force", action="store_true", help="overwrite an existing gold set")
    args = ap.parse_args()

    existing = OCR_GOLD_CSV.read_text().strip().splitlines() if OCR_GOLD_CSV.exists() else []
    if len(existing) > 1 and not args.force:
        raise SystemExit(f"{OCR_GOLD_CSV.relative_to(ROOT)} already has {len(existing) - 1} rows. Use --force to rebuild it.")

    known = {}
    labels_csv = ROOT / "data" / "brand_labels.csv"
    if labels_csv.exists():
        with open(labels_csv, newline="", encoding="utf-8") as f:
            known = {Path(r["file"]).name: r["brand"] for r in csv.DictReader(f)}

    normal = [p for p in list_images(RAW) if p.parent.name == "normal"]
    if not normal:
        raise SystemExit("No photos in data/raw/normal. Run tools/import_data.py first.")
    rng = random.Random(SEED)
    labelled = [p for p in normal if p.name in known]       # photos whose brand folder told us the brand
    rest = [p for p in normal if p.name not in known]
    pick = rng.sample(labelled, min(args.n, len(labelled)))
    if len(pick) < args.n:
        pick += rng.sample(rest, min(args.n - len(pick), len(rest)))

    gold_dir = ROOT / "data" / "ocr_gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in pick:
        shutil.copy2(p, gold_dir / p.name)
        rows.append((f"data/ocr_gold/{p.name}", known.get(p.name, "")))
    with open(OCR_GOLD_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "brand"])
        w.writerows(rows)

    REPORTS.mkdir(exist_ok=True)
    gold_paths = [gold_dir / p.name for p in pick]
    for i in range(0, len(gold_paths), 20):
        sheet(gold_paths[i:i + 20], REPORTS / f"gold_sheet_{i // 20 + 1}.png", i + 1)

    empty = sum(1 for _, b in rows if not b)
    print(f"gold set: {len(rows)} photos -> data/ocr_gold/ and {OCR_GOLD_CSV.relative_to(ROOT)}")
    print(f"brands pre-filled: {len(rows) - empty} · still empty: {empty}")
    print("contact sheets: reports/gold_sheet_*.png (row #n = line n+1 of the CSV)")
    if empty:
        print("Fill every empty brand with a name from data/brands.csv, then ask a teammate to check it.")


if __name__ == "__main__":
    main()
