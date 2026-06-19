"""Generate a synthetic predictions table + classes.json so the eval and
error-analysis flow runs locally with NO training/torch. Injects a realistic,
confident-mistake pattern (square<->triangle confusion) for the analysis to find.

Usage:  python scripts/make_sample_predictions.py --out data/run
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CLASSES = ["circle", "square", "triangle"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/run")
    ap.add_argument("--per-class", type=int, default=60)
    args = ap.parse_args()

    import pandas as pd

    rng = random.Random(0)
    rows = []
    for ci, _name in enumerate(CLASSES):
        for i in range(args.per_class):
            true = ci
            # circle easy; square/triangle confused ~25% of the time, often confidently
            if ci == 0:
                pred = 0 if rng.random() < 0.95 else rng.choice([1, 2])
            else:
                other = 2 if ci == 1 else 1
                pred = true if rng.random() < 0.75 else other
            conf = round(rng.uniform(0.85, 0.99) if pred == true else rng.uniform(0.6, 0.97), 4)
            rows.append({"path": f"data/dataset/{CLASSES[ci]}/{i:04d}.png",
                         "y_true": true, "y_pred": pred, "confidence": conf})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "predictions.parquet", index=False)
    with open(out / "classes.json", "w") as f:
        json.dump(CLASSES, f)
    print(f"Wrote {len(rows)} synthetic predictions -> {out}/predictions.parquet")


if __name__ == "__main__":
    main()
