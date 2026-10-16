"""Shared image helpers: listing, safe opening, and near-duplicate detection."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from config import CLASSES, IMAGE_EXTS, NEAR_DUP_DIST


def list_images(raw: Path) -> list[Path]:
    """All photos under raw/<class>/, sorted, any supported extension (.jpg/.jpeg/.png, any case)."""
    return sorted(
        p for p in raw.glob("*/*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.parent.name in CLASSES
    )


def label_of(path: Path) -> int:
    """1 = laptop, 0 = normal. The folder name is the label."""
    return int(path.parent.name == "laptop")


def open_rgb(path: Path) -> Image.Image:
    """Open a photo upright (phone EXIF rotation applied) as RGB."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def dhash(path: Path, size: int = 16) -> int:
    """256-bit difference hash: similar photos give hashes a few bits apart."""
    gray = open_rgb(path).convert("L").resize((size + 1, size))
    px = np.asarray(gray, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_groups(paths: list[Path], max_dist: int = NEAR_DUP_DIST) -> list[int]:
    """Group id per path; photos of the same battery (dHash distance <= max_dist) share a group."""
    hashes = [dhash(p) for p in paths]
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hamming(hashes[i], hashes[j]) <= max_dist:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(paths))]
