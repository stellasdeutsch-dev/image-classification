"""Generate a synthetic predictions table + classes.json so the eval and
error-analysis flow runs locally with NO training/torch. Injects a realistic,
confident-mistake pattern (a neighbouring-class confusion) for the analysis to find.

The class set is derived from the generated dataset (data/dataset/<class>/) when
present, so it never drifts from make_sample_dataset.py; otherwise it falls back
to the default shapes.

Usage:  python scripts/make_sample_predictions.py --out data/run
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_CLASSES = ["circle", "square", "triangle"]


def derive_classes(dataset_dir: str = "data/dataset") -> list[str]:
    p = Path(dataset_dir)
    if p.is_dir():
        dirs = sorted(d.name for d in p.iterdir() if d.is_dir())
        if dirs:
            return dirs
    return DEFAULT_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/run")
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--per-class", type=int, default=60)
    args = ap.parse_args()

    import pandas as pd

    classes = derive_classes(args.dataset)
    n = len(classes)
    rng = random.Random(0)

    def confuse(ci: int) -> int:
        if n <= 1:
            return 0
        if ci == 0:                                   # class 0 is "easy"
            return 0 if rng.random() < 0.95 else rng.randrange(1, n)
        other = ci + 1 if ci + 1 < n else ci - 1      # a neighbouring class (e.g. square<->triangle)
        return ci if rng.random() < 0.75 else other

    rows = []
    for ci in range(n):
        for i in range(args.per_class):
            pred = confuse(ci)
            conf = round(rng.uniform(0.85, 0.99) if pred == ci else rng.uniform(0.6, 0.97), 4)
            rows.append({"path": f"{args.dataset}/{classes[ci]}/{i:04d}.png",
                         "y_true": ci, "y_pred": pred, "confidence": conf})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "predictions.parquet", index=False)
    with open(out / "classes.json", "w") as f:
        json.dump(classes, f)
    print(f"Wrote {len(rows)} synthetic predictions over {n} classes -> {out}/predictions.parquet")


if __name__ == "__main__":
    main()
