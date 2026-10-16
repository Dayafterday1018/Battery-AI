"""Shared training helpers for train_cnn.py and train_transfer.py (never touches the test set)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import SEED, VAL_SIZE
from prepare import load, read_split


def load_train_val():
    """Load train split only, then hold out VAL_SIZE for validation (stratified, shuffled).

    Keras' validation_split takes the LAST 20% without shuffling; if the list is sorted by
    folder, validation would be all 'normal'. An explicit stratified split avoids that.
    """
    X, y = load(read_split("train"))
    return train_test_split(X, y, test_size=VAL_SIZE, stratify=y, random_state=SEED)


def plot_history(history, out, title):
    h = history.history
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(h["loss"], label="train"); a1.plot(h["val_loss"], label="validation")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(h["accuracy"], label="train"); a2.plot(h["val_accuracy"], label="validation")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    best = max(h["val_accuracy"])
    i = h["val_accuracy"].index(best)
    print(f"best val_accuracy {best:.2f} (train {h['accuracy'][i]:.2f}, epoch {i + 1}) · chart {out}")
