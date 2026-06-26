"""Run a trained model over the test split and write a predictions table
(path, y_true, y_pred, confidence) — the input to evaluation and error analysis.

CLI:  python -m src.predict --config configs/predict.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .common import load_config, load_json, select_device, setup_logging

log = logging.getLogger("imgcls")


def run(cfg: dict):
    import numpy as np
    import pandas as pd
    import torch

    from .data import build_eval_loader
    from .model import build_model
    from .schema import PREDICTION_COLUMNS, validate_predictions, validate_test_manifest

    device = select_device(cfg.get("device", "auto"))
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    class_names = ckpt.get("classes") or load_json(Path(cfg["out_dir"]) / "classes.json")
    model = build_model(ckpt.get("arch", cfg.get("model", "resnet18")), len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Score the exact held-out files recorded at train time (explicit contract,
    # not a re-derived seed split — so adding/removing an image can't leak).
    man = validate_test_manifest(pd.read_parquet(cfg["test_manifest_path"]))
    paths = man["path"].tolist()
    y_true = man["y_true"].to_numpy()

    loader = build_eval_loader(paths, cfg.get("image_size", 224),
                               cfg.get("batch_size", 32), cfg.get("num_workers", 4))
    preds, confs = [], []
    with torch.no_grad():
        for x in loader:                       # shuffle=False -> order matches `paths`
            probs = torch.softmax(model(x.to(device)), dim=1)
            conf, pred = probs.max(dim=1)
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())

    df = pd.DataFrame({"path": paths, "y_true": y_true,
                       "y_pred": np.concatenate(preds) if preds else np.array([], int),
                       "confidence": np.concatenate(confs) if confs else np.array([], float)},
                      columns=PREDICTION_COLUMNS)
    validate_predictions(df)
    Path(cfg["predictions_path"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg["predictions_path"], index=False)
    log.info("Wrote %d predictions -> %s", len(df), cfg["predictions_path"])
    return df


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Predict on the test split.")
    ap.add_argument("--config", default="configs/predict.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
