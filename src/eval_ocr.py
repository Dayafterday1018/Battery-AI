"""Step 3.1 validate · run the OCR on the gold set and report accuracy per brand.

data/ocr_gold.csv columns: file,brand   (file relative to repo root; brand as in brands.csv)
Run: python src/eval_ocr.py --version v1 --stage baseline
     python src/eval_ocr.py --version v2 --stage improved
     python src/eval_ocr.py --version v2 --model llava:13b --stage improved   # compare models
Writes reports/ocr_misses_<run>.csv and one metrics JSON (reports/metrics/) for the report.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime

from config import METRICS, OCR_GOLD_CSV, OLLAMA_MODEL, REPORTS, ROOT, TARGET_OCR_ACCURACY
from ocr import load_brands, read_brand_detail

CHANGES_V2 = [
    "Photo sent upright (EXIF rotation), resized to 672 px and contrast-stretched",
    "Prompt lists the allowed brands and asks for \"unknown\" instead of a guess",
    "Temperature 0, so the same photo always gets the same answer",
    "If the answer is unknown, retry the photo rotated 90° and 270°",
    "Fall back to the brand named in text_seen when the brand field is empty",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--stage", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N photos only (quick check)")
    args = ap.parse_args()

    brands = load_brands()
    with open(OCR_GOLD_CSV, newline="", encoding="utf-8") as f:
        gold = list(csv.DictReader(f))
    if args.limit:
        gold = gold[:args.limit]
    if not gold:
        raise SystemExit(f"{OCR_GOLD_CSV.relative_to(ROOT)} is empty: add ~40 labelled photos first.")

    hits, totals, misses, calls, seconds = defaultdict(int), defaultdict(int), [], 0, 0.0
    for row in gold:
        t0 = time.perf_counter()
        d = read_brand_detail(ROOT / row["file"], brands, args.version, args.model)
        seconds += time.perf_counter() - t0
        calls += d["calls"]
        totals[row["brand"]] += 1
        if d["brand"] == row["brand"]:
            hits[row["brand"]] += 1
        else:
            misses.append({"file": row["file"], "expected": row["brand"], "got": d["brand"],
                           "text_seen": str(d.get("text_seen", ""))[:120]})
        print(f"  {'ok ' if d['brand'] == row['brand'] else 'MISS'} {row['file']}: {d['brand']}", flush=True)

    n, h = sum(totals.values()), sum(hits.values())
    print()
    for b in sorted(totals, key=lambda k: -totals[k]):
        print(f"{b:<12}{hits[b]:>3}/{totals[b]}")
    verdict = "PASS" if h / n >= TARGET_OCR_ACCURACY else "below target"
    print(f"{'total':<12}{h:>3}/{n}  ({h / n:.0%}) -> {verdict} (target ≥ {TARGET_OCR_ACCURACY:.0%})")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = f"{args.version}_{args.model.replace(':', '-')}_{stamp}"
    REPORTS.mkdir(exist_ok=True); METRICS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / f"ocr_misses_{run}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "expected", "got", "text_seen"])
        w.writeheader(); w.writerows(misses)
    result = {
        "kind": "ocr", "stage": args.stage, "note": args.note, "run_at": stamp,
        "version": args.version, "model": args.model, "target": TARGET_OCR_ACCURACY,
        "gold_photos": n, "correct": h, "accuracy": h / n,
        "per_brand": {b: {"correct": hits[b], "total": totals[b]} for b in totals},
        "misses": misses, "sec_per_photo": round(seconds / n, 2), "calls_per_photo": round(calls / n, 2),
        "changes": CHANGES_V2 if args.version == "v2" else [],
    }
    (METRICS / f"ocr_{run}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{len(misses)} misses -> reports/ocr_misses_{run}.csv · metrics -> reports/metrics/ocr_{run}.json")


if __name__ == "__main__":
    main()
