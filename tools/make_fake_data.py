"""Generate FAKE battery photos to dry-run every step before Client W's real photos arrive.

laptop: dark wide pack with a yellow label and a ribbon connector.
normal: coloured cylinder with a copper cap and the BRAND printed on it, so LLaVA has text to read.

Writes (all files start with FAKE_ so they are easy to remove):
  data/raw/{laptop,normal}/   training photos              (steps 1-2)
  data/ocr_gold/ + data/ocr_gold.csv   40 normal batteries with known brand (step 3.1)
  data/inbox/ + data/inbox_expected.csv  20 mixed photos + true answer for the hand count (step 4)

Run: python tools/make_fake_data.py
Before adding real photos: delete every FAKE_* file and reset data/ocr_gold.csv to its header.
"""
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import RAW  # noqa: E402

# Brand mix of the slide-29 example gold set (40 photos)
GOLD_MIX = {"Duracell": 10, "Energizer": 9, "Panasonic": 8, "GP": 7, "Sony": 6}
BRANDS = list(GOLD_MIX)
W, H = 480, 360


def font(size):
    for name in ("Arial Bold.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def background(rng):
    return Image.new("RGB", (W, H), tuple(rng.randint(205, 250) for _ in range(3)))


def laptop(rng):
    img = background(rng)
    d = ImageDraw.Draw(img)
    x, y = rng.randint(20, 70), rng.randint(70, 150)
    w, h = rng.randint(330, 390), rng.randint(80, 120)
    d.rectangle([x, y, x + w, y + h], fill=(25, 25, 28))
    d.rectangle([x + 30, y + 20, x + 150, y + 45], fill=(225, 200, 60))      # yellow label
    for k in range(4):                                                     # cell seams
        d.line([x + (k + 1) * w // 5, y, x + (k + 1) * w // 5, y + h], fill=(45, 45, 50), width=2)
    d.rectangle([x - 18, y + 30, x, y + 60], fill=(235, 235, 235))         # ribbon connector
    return img.rotate(rng.uniform(-12, 12), fillcolor=(240, 240, 240))


def normal(rng, brand):
    """A battery lying on its side: copper cap on the left, + terminal on the right, brand in the middle."""
    img = background(rng)
    d = ImageDraw.Draw(img)
    body, ink = rng.choice([((20, 20, 20), (255, 255, 255)), ((30, 80, 190), (255, 255, 255)),
                            ((200, 40, 40), (255, 255, 255)), ((235, 235, 235), (20, 20, 20))])
    x0, x1 = rng.randint(40, 70), rng.randint(410, 440)
    y0 = rng.randint(120, 150)
    y1 = y0 + rng.randint(90, 110)
    d.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=body)
    d.rounded_rectangle([x0, y0, x0 + 80, y1], radius=14, fill=(190, 130, 60))   # copper cap
    d.rectangle([x1, (y0 + y1) // 2 - 12, x1 + 12, (y0 + y1) // 2 + 12], fill=(170, 170, 170))
    if brand:
        f = font(40 if len(brand) > 6 else 52)
        tw = d.textlength(brand.upper(), font=f)
        cx = (x0 + 80 + x1) / 2
        d.text((cx - tw / 2, (y0 + y1) / 2), brand.upper(), font=f, fill=ink, anchor="lm")
    return img.rotate(rng.uniform(-10, 10), fillcolor=(240, 240, 240))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    rng = random.Random(0)
    data = ROOT / "data"

    # Steps 1-2: training photos (brand irrelevant; ~15% of normal ones have no readable text)
    for cls in ("laptop", "normal"):
        (RAW / cls).mkdir(parents=True, exist_ok=True)
    for i in range(80):
        laptop(rng).save(RAW / "laptop" / f"FAKE_laptop_{i:03d}.jpg", quality=90)
    for i in range(120):
        brand = rng.choice(BRANDS) if rng.random() > 0.15 else None
        normal(rng, brand).save(RAW / "normal" / f"FAKE_normal_{i:03d}.jpg", quality=90)

    # Step 3.1: OCR gold set, 40 photos with known brand
    gold_dir = data / "ocr_gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold = []
    for brand, n in GOLD_MIX.items():
        for k in range(n):
            name = f"FAKE_gold_{brand.lower()}_{k:02d}.jpg"
            normal(rng, brand).save(gold_dir / name, quality=90)
            gold.append((f"data/ocr_gold/{name}", brand))
    write_csv(data / "ocr_gold.csv", ["file", "brand"], gold)

    # Step 4: inbox of 20 mixed photos + the true answer for the hand count
    inbox = data / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    expected = []
    for i in range(20):
        name = f"FAKE_inbox_{i:02d}.jpg"
        if i % 4 == 0:
            laptop(rng).save(inbox / name, quality=90)
            expected.append((name, "laptop", ""))
        else:
            brand = rng.choice(BRANDS)
            normal(rng, brand).save(inbox / name, quality=90)
            expected.append((name, "normal", brand))
    write_csv(data / "inbox_expected.csv", ["file", "label", "brand"], expected)

    print(f"raw: 80 laptop + 120 normal · gold: {len(gold)} · inbox: 20 (+ inbox_expected.csv)")


if __name__ == "__main__":
    main()
