"""Step 4 validate · compare reports/results.csv and fees.csv with a hand count.

data/inbox_expected.csv columns: file,label,brand   (brand empty for laptop)
Run after `python src/pipeline.py data/inbox/`:  python tools/compare_expected.py
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    expected = {r["file"]: r for r in rows(ROOT / "data" / "inbox_expected.csv")}
    got = {r["file"]: r for r in rows(ROOT / "reports" / "results.csv")}

    wrong = []
    for name, e in expected.items():
        g = got.get(name)
        if g is None:
            wrong.append((name, "missing from results.csv"))
        elif g["label"] != e["label"]:
            wrong.append((name, f"label {g['label']} (expected {e['label']})"))
        elif e["label"] == "normal" and g["brand"] != e["brand"]:
            wrong.append((name, f"brand {g['brand']} (expected {e['brand']})"))

    want = Counter(e["brand"] for e in expected.values() if e["label"] == "normal")
    fees = {r["brand"]: int(r["count"]) for r in rows(ROOT / "reports" / "fees.csv")}
    print(f"{'brand':<12}{'expected':>9}{'fees.csv':>10}")
    for b in sorted(set(want) | set(fees)):
        flag = "" if want.get(b, 0) == fees.get(b, 0) else "  <-- differs"
        print(f"{b:<12}{want.get(b, 0):>9}{fees.get(b, 0):>10}{flag}")
    print(f"\n{len(expected) - len(wrong)}/{len(expected)} photos correct")
    for name, why in wrong:
        print(f"  {name}: {why}")
    print("PASS" if not wrong else "NOT YET: check each photo above (unknown brands should be in review.csv)")


if __name__ == "__main__":
    main()
