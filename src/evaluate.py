"""Step 2.3 · evaluate one or more models on the test set, once. The ONLY script that reads it.

Run: python src/evaluate.py --model models/cnn.keras models/mobilenet.keras
Prints accuracy, precision and recall per class; saves a confusion matrix and the
misclassified photos per model to reports/.
"""
import argparse
import csv
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import REPORTS, ROOT, THRESHOLD
from prepare import load, read_split

NAMES = ["normal", "laptop"]  # index = label


def evaluate(model_path: Path, X, y, paths, threshold: float):
    model = keras.models.load_model(model_path)
    p = model.predict(X, verbose=0).ravel()
    pred = (p >= threshold).astype(int)
    prec, rec, _, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    name = model_path.stem

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks([0, 1], NAMES); ax.set_yticks([0, 1], NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(name)
    fig.tight_layout(); fig.savefig(REPORTS / f"{name}_confusion.png", dpi=120); plt.close(fig)

    with open(REPORTS / f"{name}_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "actual", "predicted", "p_laptop"])
        for path, a, b, pl in zip(paths, y, pred, p):
            if a != b:
                w.writerow([Path(path).relative_to(ROOT), NAMES[a], NAMES[b], f"{pl:.3f}"])

    return {
        "Test accuracy": accuracy_score(y, pred),
        "Precision, normal": prec[0], "Recall, normal": rec[0],
        "Precision, laptop": prec[1], "Recall, laptop": rec[1],
        "Errors": int((pred != y).sum()),
        "Model file (MB)": model_path.stat().st_size / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    paths = read_split("test")
    X, y = load(paths)
    results = {m.stem: evaluate(m, X, y, paths, args.threshold) for m in args.model}

    print(f"test set: {len(y)} photos · threshold {args.threshold}")
    print(f"{'metric':<20}" + "".join(f"{n:>14}" for n in results))
    for metric in next(iter(results.values())):
        print(f"{metric:<20}" + "".join(f"{r[metric]:>14.2f}" for r in results.values()))
    print("Recall, normal = share of normal batteries we catch; each miss is a fee never billed.")
    print("Misclassified photos: reports/<model>_errors.csv")


if __name__ == "__main__":
    main()
