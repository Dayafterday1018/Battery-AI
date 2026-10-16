"""Step 1 · count images per class, find corrupt and duplicate files, report sizes, save contact sheets.

Run: python src/check_images.py
"""
import hashlib
import random
from collections import defaultdict

from PIL import Image, ImageDraw

from config import CLASSES, RAW, REPORTS, SEED
from imaging import list_images, open_rgb

THUMB = 160
PER_SHEET = 20  # 5 x 4 grid


def contact_sheet(paths, out):
    cols, rows = 5, 4
    sheet = Image.new("RGB", (cols * THUMB, rows * (THUMB + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, p in enumerate(paths[:PER_SHEET]):
        img = open_rgb(p)
        img.thumbnail((THUMB, THUMB))
        x, y = (k % cols) * THUMB, (k // cols) * (THUMB + 16)
        sheet.paste(img, (x, y))
        draw.text((x + 2, y + THUMB + 2), p.name[:24], fill="black")
    sheet.save(out)


def main():
    REPORTS.mkdir(exist_ok=True)
    paths = list_images(RAW)
    by_class, corrupt, sizes, digests = defaultdict(list), [], [], defaultdict(list)

    for p in paths:
        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                sizes.append(max(img.size))
        except Exception:
            corrupt.append(p)
            continue
        by_class[p.parent.name].append(p)
        digests[hashlib.md5(p.read_bytes()).hexdigest()].append(p)

    dupes = [group for group in digests.values() if len(group) > 1]
    n_dupes = sum(len(g) - 1 for g in dupes)

    print(" · ".join(f"{c} {len(by_class[c])}" for c in CLASSES) + f" · corrupt {len(corrupt)}")
    print(f"exact duplicates {n_dupes}" + (f" · sizes {min(sizes)}-{max(sizes)} px" if sizes else ""))
    for p in corrupt:
        print(f"  corrupt: {p.relative_to(RAW)}")
    for g in dupes:
        print("  duplicate: " + " = ".join(str(p.relative_to(RAW)) for p in g))

    rng = random.Random(SEED)
    for c in CLASSES:
        if by_class[c]:
            sample = rng.sample(by_class[c], min(PER_SHEET, len(by_class[c])))
            out = REPORTS / f"contact_{c}.png"
            contact_sheet(sample, out)
            print(f"saved {out.relative_to(REPORTS.parent)}")


if __name__ == "__main__":
    main()
