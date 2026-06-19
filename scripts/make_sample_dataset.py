"""Generate a tiny folder-per-class image dataset (colored shapes) so the full
train -> predict -> eval flow runs locally without a download.

Classes = shapes {circle, square, triangle}; color varies to add some natural
confusability. Usage:
    python scripts/make_sample_dataset.py --out data/dataset --per-class 60
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle"]
COLORS = [(220, 40, 40), (40, 180, 80), (40, 90, 220), (230, 210, 40), (150, 40, 200)]


def draw(path: Path, shape: str, seed: int, size: int = 96) -> None:
    rng = random.Random(seed)
    bg = tuple(rng.randint(15, 60) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    color = rng.choice(COLORS)
    m = rng.randint(size // 6, size // 4)
    box = [m, m, size - m, size - m]
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    else:
        d.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=color)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--per-class", type=int, default=60)
    args = ap.parse_args()

    out = Path(args.out)
    for ci, shape in enumerate(SHAPES):
        d = out / shape
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            draw(d / f"{i:04d}.png", shape, seed=ci * 1000 + i)
    print(f"Wrote {len(SHAPES) * args.per_class} images across {len(SHAPES)} classes -> {out}")


if __name__ == "__main__":
    main()
