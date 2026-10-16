"""Pick the final classifier among candidates by VALIDATION accuracy (never by test accuracy).

Run: python tools/select_model.py models/mobilenet_ft40.keras models/mobilenet_ft80.keras --out models/mobilenet.keras
Copies the winner (and its .json) to --out and records why it was chosen.
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    scored = []
    for m in args.candidates:
        meta = json.loads(m.with_suffix(".json").read_text())
        scored.append((meta.get("best_val_accuracy", 0.0), m, meta))
        print(f"{m.stem:<24} validation accuracy {meta.get('best_val_accuracy', 0):.4f}")
    best_acc, best, meta = max(scored, key=lambda s: s[0])

    shutil.copy2(best, args.out)
    meta["changes"] = meta.get("changes", []) + [
        f"Chosen among {len(scored)} candidate settings by validation accuracy "
        f"({', '.join(f'{m.stem} {a:.3f}' for a, m, _ in scored)})"]
    meta["selected_from"] = best.stem
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    curves = best.parent.parent / "reports" / f"{best.stem}_curves.png"
    if curves.exists():
        shutil.copy2(curves, curves.with_name(f"{args.out.stem}_curves.png"))
    print(f"selected {best.stem} -> {args.out}")


if __name__ == "__main__":
    main()
