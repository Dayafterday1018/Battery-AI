"""Step 2.3 · evaluate one or more models on the test set, once. The ONLY script that reads it.

Run: python src/evaluate.py --model models/mobilenet_baseline.keras models/mobilenet.keras \
                            --stage baseline improved
Each model uses its own input size and threshold from models/<name>.json (default 128 px, 0.5).
Prints accuracy, precision and recall per class; saves a confusion matrix, the misclassified
photos and one metrics JSON per model (reports/metrics/, read by tools/build_report.py).
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import IMG, METRICS, REPORTS, ROOT, TARGET_CLS_ACCURACY
from prepare import load, read_split
from training import load_meta

NAMES = ["normal", "laptop"]  # index = label


def evaluate(model_path: Path, paths, threshold_override=None, cache={}):
    meta = load_meta(model_path)
    img = meta.get("img", IMG)
    threshold = threshold_override if threshold_override is not None else meta.get("threshold", 0.5)
    if img not in cache:
        cache[img] = load(paths, img)
    X, y = cache[img]

    model = keras.models.load_model(model_path)
    model.predict(X[:1], verbose=0)                       # warm-up, excluded from timing
    t0 = time.perf_counter()
    p = model.predict(X, verbose=0).ravel()
    sec_per_photo = (time.perf_counter() - t0) / len(X)
    pred = (p >= threshold).astype(int)
    prec, rec, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    name = model_path.stem

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks([0, 1], NAMES); ax.set_yticks([0, 1], NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(name)
    fig.tight_layout(); fig.savefig(REPORTS / f"{name}_confusion.png", dpi=120); plt.close(fig)

    errors = [{"file": str(Path(f).relative_to(ROOT)), "actual": NAMES[a], "predicted": NAMES[b],
               "p_laptop": round(float(pl), 3)}
              for f, a, b, pl in zip(paths, y, pred, p) if a != b]
    with open(REPORTS / f"{name}_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "actual", "predicted", "p_laptop"])
        w.writeheader(); w.writerows(errors)

    return {
        "model": name, "family": meta.get("family", ""), "img": img, "threshold": threshold,
        "test_photos": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_normal": float(prec[0]), "recall_normal": float(rec[0]),
        "precision_laptop": float(prec[1]), "recall_laptop": float(rec[1]),
        "confusion": {"normal_as_normal": int(cm[0, 0]), "normal_as_laptop": int(cm[0, 1]),
                      "laptop_as_normal": int(cm[1, 0]), "laptop_as_laptop": int(cm[1, 1])},
        "errors": errors,
        "model_mb": round(model_path.stat().st_size / 1e6, 2),
        "sec_per_photo_cpu": round(sec_per_photo, 4),
        "best_val_accuracy": meta.get("best_val_accuracy"),
        "train_accuracy_at_best": meta.get("train_accuracy_at_best"),
        "changes": meta.get("changes", []),
        "settings": meta.get("settings", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", type=Path, required=True)
    ap.add_argument("--stage", nargs="*", default=[], help="label per model, e.g. baseline improved")
    ap.add_argument("--note", default="", help="free text saved with the run")
    ap.add_argument("--threshold", type=float, default=None, help="override every model's threshold")
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True); METRICS.mkdir(parents=True, exist_ok=True)

    paths = read_split("test")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []
    for i, m in enumerate(args.model):
        r = evaluate(m, paths, args.threshold)
        r.update(kind="classifier", stage=args.stage[i] if i < len(args.stage) else "",
                 note=args.note, run_at=stamp, target=TARGET_CLS_ACCURACY)
        (METRICS / f"cls_{r['model']}_{stamp}.json").write_text(json.dumps(r, indent=2))
        results.append(r)

    print(f"test set: {results[0]['test_photos']} photos · target accuracy ≥ {TARGET_CLS_ACCURACY:.0%}")
    rows = [("Test accuracy", "accuracy"), ("Recall, normal", "recall_normal"),
            ("Recall, laptop", "recall_laptop"), ("Precision, normal", "precision_normal"),
            ("Precision, laptop", "precision_laptop"), ("Threshold", "threshold"),
            ("Input px", "img"), ("Model file (MB)", "model_mb"), ("Sec/photo (CPU)", "sec_per_photo_cpu")]
    print(f"{'metric':<20}" + "".join(f"{r['model'][:16]:>18}" for r in results))
    for label, key in rows:
        print(f"{label:<20}" + "".join(f"{r[key]:>18.3f}" for r in results))
    for r in results:
        verdict = "PASS" if r["accuracy"] >= TARGET_CLS_ACCURACY else "below target"
        print(f"{r['model']}: {r['accuracy']:.1%} -> {verdict} · {len(r['errors'])} errors in reports/{r['model']}_errors.csv")
    print(f"metrics saved to {METRICS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
