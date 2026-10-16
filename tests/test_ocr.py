import httpx
import pytest

import ocr

BRANDS = ["Duracell", "Energizer", "Panasonic", "GP", "Sony"]


@pytest.mark.parametrize("raw, expected", [
    ("DURACELL", "Duracell"),
    ("duracell plus", "Duracell"),
    ("GP", "GP"),
    ("gp", "GP"),                 # the slide version (.title() + case-sensitive match) misses this
    ("Panasonlc", "Panasonic"),   # one OCR typo
    ("unknown", "unknown"),
    ("", "unknown"),
    ("Acme Power", "unknown"),
])
def test_match_brand(raw, expected):
    assert ocr.match_brand(raw, BRANDS) == expected


class FakeResponse:
    def __init__(self, text):
        self._text = text
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._text}


def test_read_brand_parses_llava_json(monkeypatch, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")
    monkeypatch.setattr(ocr.httpx, "post",
                        lambda *a, **k: FakeResponse('{"brand": "ENERGIZER", "text_seen": "ENERGIZER MAX"}'))
    assert ocr.read_brand(img, BRANDS) == "Energizer"


def test_read_brand_non_json_is_unknown(monkeypatch, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")
    monkeypatch.setattr(ocr.httpx, "post", lambda *a, **k: FakeResponse("The brand is Duracell."))
    assert ocr.read_brand(img, BRANDS) == "unknown"


def test_clear_error_when_ollama_down(monkeypatch, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"fake")

    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ocr.httpx, "post", down)
    with pytest.raises(ocr.OllamaUnavailable, match="ollama serve"):
        ocr.read_brand(img, BRANDS)
