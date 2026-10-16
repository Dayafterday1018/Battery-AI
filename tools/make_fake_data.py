"""Generate FAKE battery photos to dry-run the whole pipeline before Client W's photos arrive.

laptop: dark wide rectangle with a light ribbon strip  ·  normal: coloured upright cylinder.
Run: python tools/make_fake_data.py [--n 80]   -> data/raw/{laptop,normal}, data/inbox/
Delete data/raw/* before adding real photos.
"""
import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import RAW  # noqa: E402


def laptop(rng):
    img = Image.new("RGB", (480, 360), tuple(rng.randint(200, 255) for _ in range(3)))
    d = ImageDraw.Draw(img)
    x, y = rng.randint(20, 80), rng.randint(60, 140)
    d.rectangle([x, y, x + rng.randint(300, 380), y + rng.randint(80, 130)], fill=(25, 25, 28))
    d.rectangle([x + 20, y + 20, x + 120, y + 40], fill=(220, 200, 60))   # yellow label
    d.rectangle([x - 15, y + 30, x, y + 60], fill=(230, 230, 230))      # ribbon connector
    return img.rotate(rng.uniform(-15, 15), fillcolor=(240, 240, 240))


def normal(rng):
    img = Image.new("RGB", (480, 360), tuple(rng.randint(200, 255) for _ in range(3)))
    d = ImageDraw.Draw(img)
    body = rng.choice([(184, 115, 51), (20, 20, 20), (30, 90, 200), (200, 40, 40)])
    x, w = rng.randint(150, 260), rng.randint(70, 110)
    d.rectangle([x, 60, x + w, 320], fill=body)
    d.rectangle([x, 60, x + w, 110], fill=(190, 140, 70))               # copper top
    d.rectangle([x + w // 3, 45, x + 2 * w // 3, 60], fill=(170, 170, 170))  # + terminal
    return img.rotate(rng.uniform(-20, 20), fillcolor=(240, 240, 240))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()
    rng = random.Random(0)
    for cls, draw in (("laptop", laptop), ("normal", normal)):
        (RAW / cls).mkdir(parents=True, exist_ok=True)
        for i in range(args.n if cls == "laptop" else int(args.n * 1.5)):
            draw(rng).save(RAW / cls / f"FAKE_{cls}_{i:03d}.jpg", quality=90)
    inbox = ROOT / "data" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (laptop if i % 3 == 0 else normal)(rng).save(inbox / f"FAKE_inbox_{i:02d}.jpg")
    print(f"fake photos written to {RAW} and {inbox}")


if __name__ == "__main__":
    main()
