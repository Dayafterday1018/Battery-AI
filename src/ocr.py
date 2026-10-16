"""Step 3.1 · read the brand printed on a normal battery with LLaVA (local, through Ollama).

Run: ollama serve            (in another terminal)
     python src/ocr.py <image>
Returns a brand from data/brands.csv, or "unknown" (which goes to manual review, never a bill).
"""
import base64
import csv
import difflib
import json
import sys
from pathlib import Path

import httpx

from config import BRAND_CUTOFF, BRANDS_CSV, OLLAMA, OLLAMA_MODEL, OLLAMA_TIMEOUT

PROMPT = ("Read the text printed on this battery. "
          'Reply only with JSON: {"brand": "...", "text_seen": "..."}. '
          'Use "unknown" if you cannot read a brand.')


class OllamaUnavailable(RuntimeError):
    pass


def load_brands(path: Path = BRANDS_CSV) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row["brand"].strip() for row in csv.DictReader(f) if row["brand"].strip()]


def match_brand(raw: str, brands: list[str]) -> str:
    """Fuzzy, case-insensitive: 'DURACELL' -> 'Duracell', 'gp' -> 'GP', 'Panasonlc' -> 'Panasonic'."""
    raw = (raw or "").strip().lower()
    if not raw or raw == "unknown":
        return "unknown"
    lower = {b.lower(): b for b in brands}
    hit = difflib.get_close_matches(raw, lower, n=1, cutoff=BRAND_CUTOFF)
    if hit:
        return lower[hit[0]]
    words = raw.replace(",", " ").split()
    for key, brand in lower.items():          # brand inside a longer string: "duracell plus power"
        if key in words or raw.startswith(key + " "):
            return brand
    return "unknown"


def ask_llava(path: Path) -> dict:
    img = base64.b64encode(Path(path).read_bytes()).decode()
    try:
        r = httpx.post(f"{OLLAMA}/api/generate", timeout=OLLAMA_TIMEOUT, json={
            "model": OLLAMA_MODEL, "prompt": PROMPT, "images": [img],
            "format": "json", "stream": False})
        r.raise_for_status()
    except httpx.ConnectError as e:
        raise OllamaUnavailable(
            f"Cannot reach Ollama at {OLLAMA}. Start it with `ollama serve` "
            f"and pull the model with `ollama pull {OLLAMA_MODEL}`.") from e
    try:
        return json.loads(r.json()["response"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"brand": "unknown", "text_seen": r.text[:200]}


def read_brand(path, brands: list[str] | None = None) -> str:
    answer = ask_llava(Path(path))
    return match_brand(str(answer.get("brand", "")), brands or load_brands())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python src/ocr.py <image>")
    answer = ask_llava(Path(sys.argv[1]))
    print(f"llava said: {answer}")
    print(f"brand: {match_brand(str(answer.get('brand', '')), load_brands())}")
