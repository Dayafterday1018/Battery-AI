"""Copy the class photos into data/raw/laptop and data/raw/normal, whatever their original layout.

Auto-detect: finds folders named like laptop* and normal* (or other*) anywhere under --src.
Explicit:    --laptop DIR --normal DIR  when the folders have other names.
Brand labels: if a photo sits in a folder named after a brand in data/brands.csv
             (e.g. .../normal/Duracell/x.jpg), the brand is written to data/brand_labels.csv;
             tools/make_gold_template.py then pre-fills the OCR gold set with it.

Run: python tools/import_data.py --src /path/to/class/images
     python tools/import_data.py --laptop /path/A --normal /path/B
     add --link to symlink instead of copying (saves disk space)
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import BRANDS_CSV, IMAGE_EXTS, RAW  # noqa: E402

ALIASES = {"laptop": ("laptop", "notebook"), "normal": ("normal", "other", "non_laptop", "non-laptop", "nonlaptop")}


def find_class_dirs(src: Path) -> dict:
    found = {"laptop": [], "normal": []}
    for d in [src, *[p for p in src.rglob("*") if p.is_dir()]]:
        name = d.name.lower()
        for cls, names in ALIASES.items():
            if any(name == n or name.startswith(n) for n in names):
                # skip nested matches such as normal/laptop_lookalike inside an already matched folder
                if not any(parent in found[c] for c in found for parent in d.parents):
                    found[cls].append(d)
    return found


def images_under(d: Path):
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and not p.name.startswith("."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path)
    ap.add_argument("--laptop", type=Path, nargs="*", default=[])
    ap.add_argument("--normal", type=Path, nargs="*", default=[])
    ap.add_argument("--link", action="store_true", help="symlink instead of copy")
    args = ap.parse_args()

    dirs = {"laptop": list(args.laptop), "normal": list(args.normal)}
    if args.src:
        auto = find_class_dirs(args.src.resolve())
        for cls in dirs:
            dirs[cls] = dirs[cls] or auto[cls]
    for cls, ds in dirs.items():
        if not ds:
            raise SystemExit(f"No '{cls}' folder found. Pass it explicitly: --{cls} /path/to/folder")
        print(f"{cls:<7} <- " + ", ".join(str(d) for d in ds))

    brands = {}
    if BRANDS_CSV.exists():
        with open(BRANDS_CSV, newline="", encoding="utf-8") as f:
            brands = {r["brand"].strip().lower(): r["brand"].strip() for r in csv.DictReader(f)}

    labels, counts = [], {}
    for cls, ds in dirs.items():
        out = RAW / cls
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for d in ds:
            for p in images_under(d):
                target = out / p.name
                k = 1
                while target.exists() or target.is_symlink():
                    target = out / f"{p.stem}_{k}{p.suffix}"
                    k += 1
                if args.link:
                    os.symlink(p.resolve(), target)
                else:
                    shutil.copy2(p, target)
                n += 1
                brand = next((brands[a.name.lower()] for a in p.parents if a.name.lower() in brands), "")
                if cls == "normal" and brand:
                    labels.append((str(target.relative_to(ROOT)), brand))
        counts[cls] = n

    print(f"imported laptop {counts['laptop']} · normal {counts['normal']} -> {RAW.relative_to(ROOT)}/")
    if labels:
        with open(ROOT / "data" / "brand_labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "brand"])
            w.writerows(labels)
        print(f"brand labels found for {len(labels)} normal photos -> data/brand_labels.csv")
    print("next: python src/check_images.py")


if __name__ == "__main__":
    main()
