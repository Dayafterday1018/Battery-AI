"""Step 3.1 validate · run read_brand on the gold set and report accuracy per brand.

data/ocr_gold.csv columns: file,brand   (file relative to repo root; brand as in brands.csv)
Run: python src/eval_ocr.py
"""
import csv
from collections import defaultdict

from config import OCR_GOLD_CSV, REPORTS, ROOT
from ocr import load_brands, read_brand


def main():
    brands = load_brands()
    with open(OCR_GOLD_CSV, newline="", encoding="utf-8") as f:
        gold = list(csv.DictReader(f))
    if not gold:
        raise SystemExit(f"{OCR_GOLD_CSV.relative_to(ROOT)} is empty: add ~40 labelled photos first.")

    hits, totals, misses = defaultdict(int), defaultdict(int), []
    for row in gold:
        got = read_brand(ROOT / row["file"], brands)
        totals[row["brand"]] += 1
        if got == row["brand"]:
            hits[row["brand"]] += 1
        else:
            misses.append((row["file"], row["brand"], got))

    for b in sorted(totals, key=lambda k: -totals[k]):
        print(f"{b:<12}{hits[b]:>3}/{totals[b]}")
    n, h = sum(totals.values()), sum(hits.values())
    print(f"{'total':<12}{h:>3}/{n}  ({h / n:.0%})")

    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "ocr_misses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "expected", "got"])
        w.writerows(misses)
    print(f"{len(misses)} misses -> reports/ocr_misses.csv (open each: label turned away? logo only?)")


if __name__ == "__main__":
    main()
