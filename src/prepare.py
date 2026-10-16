"""Step 2.1 · split data/raw into 80% train / 20% test and reshape photos to IMG x IMG x 3.

The split is stratified by label AND grouped by near-duplicate, so two shots of the same
battery never end up on both sides (that would leak the answer into the test set).

Run:  python src/prepare.py           # write data/splits/train.txt and test.txt
      python src/prepare.py --check   # verify ratios, overlap, near-duplicates and shapes
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from config import IMG, NEAR_DUP_DIST, RAW, ROOT, SEED, SPLITS, TEST_SIZE
from imaging import dhash, hamming, label_of, list_images, near_duplicate_groups, open_rgb


def split(raw: Path = RAW, out: Path = SPLITS):
    paths = list_images(raw)
    labels = [label_of(p) for p in paths]
    groups = near_duplicate_groups(paths)
    n_splits = round(1 / TEST_SIZE)  # 5 folds -> one fold = 20% test
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(cv.split(paths, labels, groups))
    out.mkdir(parents=True, exist_ok=True)
    save_list([paths[i] for i in train_idx], out / "train.txt")
    save_list([paths[i] for i in test_idx], out / "test.txt")
    return [paths[i] for i in train_idx], [paths[i] for i in test_idx]


def save_list(paths, file: Path):
    file.write_text("\n".join(str(p.relative_to(ROOT)) for p in paths) + "\n")


def read_split(name: str, splits: Path = SPLITS) -> list[Path]:
    """name = 'train' or 'test'. Only src/evaluate.py may read 'test'."""
    lines = (splits / f"{name}.txt").read_text().split()
    return [ROOT / line for line in lines]


def load(paths, size: int = IMG) -> tuple[np.ndarray, np.ndarray]:
    """Reshape: any photo -> (n, size, size, 3), pixels scaled to 0..1. y: 1 = laptop."""
    X = np.stack([
        np.asarray(open_rgb(p).resize((size, size)), dtype=np.float32) for p in paths
    ]) / 255.0
    y = np.array([label_of(Path(p)) for p in paths], dtype=np.int32)
    return X, y


def check(splits: Path = SPLITS):
    train, test = read_split("train", splits), read_split("test", splits)
    for name, ps in (("train", train), ("test", test)):
        laptop = sum(label_of(p) for p in ps) / max(len(ps), 1)
        print(f"{name:<6}{len(ps):>5}  laptop {laptop:.0%} · normal {1 - laptop:.0%}")
    same = set(train) & set(test)
    print(f"same file in both sets: {len(same)}")
    h_train = {p: dhash(p) for p in train}
    h_test = {p: dhash(p) for p in test}
    pairs = [(a, b) for b, hb in h_test.items() for a, ha in h_train.items()
             if hamming(ha, hb) <= NEAR_DUP_DIST]
    print(f"near-duplicates across sets: {len(pairs)}")
    for a, b in pairs[:10]:
        print(f"  {a.name} ~ {b.name}")
    X, _ = load(train[:8])
    print(f"X_train ({len(train)}, {', '.join(map(str, X.shape[1:]))})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        tr, te = split()
        print(f"train {len(tr)} · test {len(te)} -> {SPLITS.relative_to(ROOT)}/")
