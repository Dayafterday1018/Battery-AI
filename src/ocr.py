"""Step 3.1 · read the brand printed on a normal battery with a local vision model (Ollama).

Two versions, so the report can compare them on the same gold set:
  v1  last week: raw photo, open question ("read the text"), fuzzy match to data/brands.csv
  v2  improved:  photo upright, resized and contrast-stretched; the prompt lists the allowed
                 brands; temperature 0; if the answer is "unknown", retry the photo rotated
                 90° and 270° (printed text on a battery often runs along the cell)

Run: ollama serve            (in another terminal)
     python src/ocr.py <image> [--version v1|v2] [--model llava]
Returns a brand from data/brands.csv, or "unknown" (which goes to manual review, never a bill).
"""
import argparse
import base64
import csv
import difflib
import io
import json
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from config import BRAND_CUTOFF, BRANDS_CSV, OLLAMA, OLLAMA_MODEL, OLLAMA_TIMEOUT

PROMPT_V1 = ("Read the text printed on this battery. "
             'Reply only with JSON: {"brand": "...", "text_seen": "..."}. '
             'Use "unknown" if you cannot read a brand.')

PROMPT_V2 = ("You are reading the manufacturer brand printed on a single battery.\n"
             "Allowed brands: {brands}.\n"
             "Look at the printed text and logo. Answer with the allowed brand that appears on the battery, "
             'or "unknown" if none of them is visible or you are not sure. Do not guess.\n'
             'Reply only with JSON: {{"brand": "...", "text_seen": "..."}}')

MAX_SIDE = 672          # LLaVA 1.6 tiles images at 336 px; larger photos only add latency
ROTATIONS_V2 = (0, 90, 270)


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


def encode(path: Path, rotate: int = 0, enhance: bool = False) -> str:
    """v1 sends the file as is; v2 sends it upright, resized and contrast-stretched."""
    if not enhance:
        return base64.b64encode(Path(path).read_bytes()).decode()
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    img = ImageOps.autocontrast(img, cutoff=1)
    if rotate:
        img = img.rotate(rotate, expand=True)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def ask_model(image_b64: str, prompt: str, model: str = OLLAMA_MODEL, deterministic: bool = False) -> dict:
    payload = {"model": model, "prompt": prompt, "images": [image_b64], "format": "json", "stream": False}
    if deterministic:
        payload["options"] = {"temperature": 0}
    try:
        r = httpx.post(f"{OLLAMA}/api/generate", timeout=OLLAMA_TIMEOUT, json=payload)
        r.raise_for_status()
    except httpx.ConnectError as e:
        raise OllamaUnavailable(
            f"Cannot reach Ollama at {OLLAMA}. Start it with `ollama serve` "
            f"and pull the model with `ollama pull {model}`.") from e
    try:
        return json.loads(r.json()["response"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"brand": "unknown", "text_seen": r.text[:200]}


def read_brand_detail(path, brands: list[str] | None = None, version: str = "v2",
                      model: str = OLLAMA_MODEL) -> dict:
    """Brand plus what the model saw and how many calls it took (for the report)."""
    brands = brands or load_brands()
    if version == "v1":
        answer = ask_model(encode(path), PROMPT_V1, model)
        return {"brand": match_brand(str(answer.get("brand", "")), brands),
                "raw": answer.get("brand", ""), "text_seen": answer.get("text_seen", ""), "calls": 1}

    prompt = PROMPT_V2.format(brands=", ".join(brands))
    answer, calls = {}, 0
    for rotate in ROTATIONS_V2:
        answer = ask_model(encode(path, rotate, enhance=True), prompt, model, deterministic=True)
        calls += 1
        brand = match_brand(str(answer.get("brand", "")), brands)
        if brand == "unknown":                     # the text_seen may still name a brand
            brand = match_brand(str(answer.get("text_seen", "")), brands)
        if brand != "unknown":
            return {"brand": brand, "raw": answer.get("brand", ""),
                    "text_seen": answer.get("text_seen", ""), "calls": calls, "rotation": rotate}
    return {"brand": "unknown", "raw": answer.get("brand", ""),
            "text_seen": answer.get("text_seen", ""), "calls": calls}


def read_brand(path, brands: list[str] | None = None, version: str = "v2", model: str = OLLAMA_MODEL) -> str:
    return read_brand_detail(path, brands, version, model)["brand"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    args = ap.parse_args()
    print(read_brand_detail(args.image, version=args.version, model=args.model))
