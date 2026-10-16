"""Pipeline tests with the classifier and Ollama mocked (no TensorFlow, no LLaVA needed)."""
import csv

import numpy as np
from PIL import Image

import pipeline


class FakeModel:
    """Returns p(laptop) by file order: names starting with 'lap' are laptops."""

    def __init__(self, names):
        self.p = [0.9 if n.startswith("lap") else 0.1 for n in names]
        self.i = 0

    def predict(self, X, verbose=0):
        assert X.shape[1:] == (128, 128, 3)
        p = self.p[self.i]
        self.i += 1
        return np.array([[p]])


def make_photos(folder, names):
    folder.mkdir()
    for n in names:
        Image.new("RGB", (300, 200), "gray").save(folder / n)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))[1:]


def test_pipeline_fees_and_review(tmp_path):
    names = ["lap1.jpg", "n1.jpg", "n2.JPG", "n3.png", "n4.jpeg", "notes.txt"]
    inbox = tmp_path / "inbox"
    make_photos(inbox, [n for n in names if not n.endswith(".txt")])
    (inbox / "notes.txt").write_text("ignore me")
    brands = {"n1.jpg": "Duracell", "n2.JPG": "Duracell", "n3.png": "GP", "n4.jpeg": "unknown"}
    photos = sorted(n for n in names if not n.endswith(".txt"))

    rows, fees, review = pipeline.run(inbox, FakeModel(photos), lambda p: brands[p.name],
                                      threshold=0.5, out=tmp_path / "reports")

    assert len(rows) == 5                                   # .JPG/.png/.jpeg included, .txt skipped
    assert fees == {"Duracell": 2, "GP": 1}                 # laptop and unknown never billed
    assert review == ["n4.jpeg"]
    assert read_rows(tmp_path / "reports" / "fees.csv") == [["Duracell", "2"], ["GP", "1"]]
    assert ["lap1.jpg", "laptop", "0.9", ""] in read_rows(tmp_path / "reports" / "results.csv")


def test_laptop_never_calls_ocr(tmp_path):
    inbox = tmp_path / "inbox"
    make_photos(inbox, ["lap1.jpg", "lap2.jpg"])

    def boom(_):
        raise AssertionError("OCR called on a laptop battery")

    rows, fees, review = pipeline.run(inbox, FakeModel(["lap1.jpg", "lap2.jpg"]), boom,
                                      out=tmp_path / "reports")
    assert not fees and not review
