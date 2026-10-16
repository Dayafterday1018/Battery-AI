import json

import httpx
import pytest
from PIL import Image

import ocr

BRANDS = ["Duracell", "Energizer", "Panasonic", "GP", "Sony"]


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (1200, 800), "gray").save(p)
    return p


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


class FakeOllama:
    """Returns the queued answers in order and records each request."""

    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, url, timeout, json):
        self.requests.append(json)
        return FakeResponse(self.answers.pop(0))


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self.text}


def test_v1_parses_json(monkeypatch, img):
    fake = FakeOllama('{"brand": "ENERGIZER", "text_seen": "ENERGIZER MAX"}')
    monkeypatch.setattr(ocr.httpx, "post", fake)
    assert ocr.read_brand(img, BRANDS, version="v1") == "Energizer"
    assert "Allowed brands" not in fake.requests[0]["prompt"]


def test_v2_prompt_lists_brands_and_is_deterministic(monkeypatch, img):
    fake = FakeOllama('{"brand": "Sony", "text_seen": "SONY"}')
    monkeypatch.setattr(ocr.httpx, "post", fake)
    assert ocr.read_brand(img, BRANDS) == "Sony"
    req = fake.requests[0]
    assert "Duracell, Energizer, Panasonic, GP, Sony" in req["prompt"]
    assert req["options"] == {"temperature": 0}


def test_v2_resizes_photo(monkeypatch, img):
    fake = FakeOllama('{"brand": "GP", "text_seen": "GP"}')
    monkeypatch.setattr(ocr.httpx, "post", fake)
    ocr.read_brand(img, BRANDS)
    import base64, io
    sent = Image.open(io.BytesIO(base64.b64decode(fake.requests[0]["images"][0])))
    assert max(sent.size) == ocr.MAX_SIDE


def test_v2_retries_rotated_when_unknown(monkeypatch, img):
    fake = FakeOllama('{"brand": "unknown", "text_seen": ""}',
                      '{"brand": "Panasonic", "text_seen": "PANASONIC"}')
    monkeypatch.setattr(ocr.httpx, "post", fake)
    d = ocr.read_brand_detail(img, BRANDS)
    assert d["brand"] == "Panasonic" and d["calls"] == 2 and d["rotation"] == 90


def test_v2_uses_text_seen_when_brand_empty(monkeypatch, img):
    fake = FakeOllama('{"brand": "", "text_seen": "DURACELL PLUS 1.5V"}')
    monkeypatch.setattr(ocr.httpx, "post", fake)
    assert ocr.read_brand(img, BRANDS) == "Duracell"


def test_v2_gives_up_after_three_rotations(monkeypatch, img):
    fake = FakeOllama(*['{"brand": "unknown", "text_seen": ""}'] * 3)
    monkeypatch.setattr(ocr.httpx, "post", fake)
    d = ocr.read_brand_detail(img, BRANDS)
    assert d["brand"] == "unknown" and d["calls"] == 3


def test_non_json_is_unknown(monkeypatch, img):
    monkeypatch.setattr(ocr.httpx, "post", FakeOllama("The brand is Duracell."))
    assert ocr.read_brand(img, BRANDS, version="v1") == "unknown"


def test_clear_error_when_ollama_down(monkeypatch, img):
    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ocr.httpx, "post", down)
    with pytest.raises(ocr.OllamaUnavailable, match="ollama serve"):
        ocr.read_brand(img, BRANDS)
