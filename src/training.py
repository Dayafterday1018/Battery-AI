"""Shared training helpers (never touch the test set)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from config import IMG, SEED, TARGET_NORMAL_RECALL, VAL_SIZE
from prepare import load, read_split


def load_train_val(size: int = IMG):
    """Load train split only, then hold out VAL_SIZE for validation (stratified, shuffled).

    Keras' validation_split takes the LAST 20% without shuffling; if the list is sorted by
    folder, validation would be all 'normal'. An explicit stratified split avoids that.
    """
    X, y = load(read_split("train"), size)
    return train_test_split(X, y, test_size=VAL_SIZE, stratify=y, random_state=SEED)


def class_weights(y) -> dict:
    """Balance laptop / normal so the minority class counts as much as the majority."""
    n, pos = len(y), int(np.sum(y))
    return {0: n / (2 * (n - pos)), 1: n / (2 * pos)}


def tune_threshold(p_laptop, y_val, target_recall: float = TARGET_NORMAL_RECALL) -> float:
    """Pick the p(laptop) cut-off on VALIDATION data (never test).

    Highest validation accuracy among thresholds that keep normal recall >= target;
    ties go to the threshold closest to 0.5.
    """
    best, best_key = 0.5, (-1.0, 0.0)
    for t in np.round(np.linspace(0.05, 0.95, 91), 2):
        pred = (p_laptop >= t).astype(int)
        normal = y_val == 0
        recall_normal = float(np.mean(pred[normal] == 0)) if normal.any() else 1.0
        if recall_normal < target_recall:
            continue
        key = (float(np.mean(pred == y_val)), -abs(t - 0.5))
        if key > best_key:
            best, best_key = float(t), key
    return best


def save_meta(model_path: Path, **meta):
    """Sidecar models/<name>.json: input size, threshold and the settings used to train."""
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_meta(model_path: Path) -> dict:
    side = Path(model_path).with_suffix(".json")
    return json.loads(side.read_text()) if side.exists() else {}


def plot_history(histories, out, title):
    """histories: one History or a list (e.g. head training + fine-tuning), drawn end to end."""
    if not isinstance(histories, list):
        histories = [histories]
    h = {k: sum((hist.history[k] for hist in histories), []) for k in
         ("loss", "val_loss", "accuracy", "val_accuracy")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(h["loss"], label="train"); a1.plot(h["val_loss"], label="validation")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(h["accuracy"], label="train"); a2.plot(h["val_accuracy"], label="validation")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    if len(histories) > 1:
        for ax in (a1, a2):
            ax.axvline(len(histories[0].history["loss"]) - 0.5, color="grey", ls="--", lw=1)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    best = max(h["val_accuracy"])
    i = h["val_accuracy"].index(best)
    print(f"best val_accuracy {best:.3f} (train {h['accuracy'][i]:.3f}, epoch {i + 1}) · chart {out}")
    return {"best_val_accuracy": best, "train_accuracy_at_best": h["accuracy"][i], "epochs": len(h["loss"])}
