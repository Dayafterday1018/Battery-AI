"""Step 4 · full operation: classify each photo; laptop -> skip; normal -> read the brand.

Run: python src/pipeline.py data/inbox/
Writes reports/results.csv (file, label, confidence, brand)
       reports/fees.csv    (brand, count)   billable brands only
       reports/review.csv  (file)           normal batteries whose brand is "unknown"
"""
import csv
import sys
from collections import Counter
from pathlib import Path

from config import IMAGE_EXTS, MODEL_PATH, REPORTS, THRESHOLD
from prepare import load


def load_classifier(path: Path = MODEL_PATH):
    import keras  # imported here so tests can run without TensorFlow
    if not Path(path).exists():
        raise FileNotFoundError(f"No model at {path}. Train one (step 2) or set BATTERY_MODEL.")
    return keras.models.load_model(path)


def run(folder, model, read_brand, threshold: float = THRESHOLD, out: Path = REPORTS):
    photos = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rows = []
    for path in photos:
        X, _ = load([path])                                   # reshape
        p = float(model.predict(X, verbose=0)[0, 0])          # p(laptop)
        if p >= threshold:
            rows.append((path.name, "laptop", round(p, 3), ""))  # no fee
            continue
        rows.append((path.name, "normal", round(1 - p, 3), read_brand(path)))  # LLaVA OCR

    billable = [r[3] for r in rows if r[1] == "normal"]
    fees = Counter(b for b in billable if b != "unknown")
    review = [r[0] for r in rows if r[1] == "normal" and r[3] == "unknown"]

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "results.csv", ["file", "label", "confidence", "brand"], rows)
    write_csv(out / "fees.csv", ["brand", "count"], fees.most_common())
    write_csv(out / "review.csv", ["file"], [(f,) for f in review])
    return rows, fees, review


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python src/pipeline.py <folder of photos>")
    from ocr import load_brands, read_brand

    brands = load_brands()
    rows, fees, review = run(sys.argv[1], load_classifier(), lambda p: read_brand(p, brands))
    laptops = sum(r[1] == "laptop" for r in rows)
    print(f"{len(rows)} photos · laptop {laptops} · normal {len(rows) - laptops} · review {len(review)}")
    for brand, n in fees.most_common():
        print(f"  {brand:<12}{n}")
    print("-> reports/results.csv, reports/fees.csv, reports/review.csv")
