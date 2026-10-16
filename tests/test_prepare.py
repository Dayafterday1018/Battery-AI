"""The split must keep near-duplicates (same battery shot twice) on one side, with equal class ratios."""
import numpy as np
from PIL import Image

import prepare
from imaging import near_duplicate_groups


def blob(path, seed, shift=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    arr = np.roll(arr, shift, axis=1)
    Image.fromarray(arr).resize((256, 256)).save(path)


def test_near_duplicates_grouped(tmp_path):
    a, b, c = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"
    blob(a, 1)
    Image.open(a).resize((250, 250)).save(b)    # same battery, slightly different shot
    blob(c, 2)
    g = near_duplicate_groups([a, b, c])
    assert g[0] == g[1] != g[2]


def test_split_keeps_pairs_together(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    for cls, n in (("laptop", 20), ("normal", 30)):
        (raw / cls).mkdir(parents=True)
        for i in range(n):
            blob(raw / cls / f"{cls}_{i}.jpg", seed=i + (0 if cls == "laptop" else 1000))
        # a second shot of battery 0 in each class
        Image.open(raw / cls / f"{cls}_0.jpg").resize((240, 240)).save(raw / cls / f"{cls}_0b.jpg")

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    train, test = prepare.split(raw, tmp_path / "splits")
    names_train = {p.name for p in train}
    for cls in ("laptop", "normal"):
        assert (f"{cls}_0.jpg" in names_train) == (f"{cls}_0b.jpg" in names_train)
    assert not set(train) & set(test)
    assert 0.15 <= len(test) / (len(train) + len(test)) <= 0.25


def test_load_shape(tmp_path):
    (tmp_path / "laptop").mkdir()
    p = tmp_path / "laptop" / "x.png"
    Image.new("RGB", (640, 480), "red").save(p)
    X, y = prepare.load([p])
    assert X.shape == (1, 128, 128, 3) and X.max() <= 1.0 and y.tolist() == [1]
