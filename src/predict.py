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
    from torch.utils.data import DataLoader

    from .data import build_datasets
    from .model import build_model

    device = select_device(cfg.get("device", "auto"))
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    class_names = ckpt.get("classes") or load_json(Path(cfg["out_dir"]) / "classes.json")
    model = build_model(ckpt.get("arch", cfg.get("model", "resnet18")), len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, _, test_ds, _ = build_datasets(cfg)
    # recover file paths for the subset (ImageFolder samples are (path, label))
    base = test_ds.dataset
    paths = [base.samples[i][0] for i in test_ds.indices]

    dl = DataLoader(test_ds, batch_size=cfg.get("batch_size", 32), shuffle=False,
                    num_workers=cfg.get("num_workers", 4))
    preds, trues, confs = [], [], []
    with torch.no_grad():
        for x, y in dl:
            probs = torch.softmax(model(x.to(device)), dim=1)
            conf, pred = probs.max(dim=1)
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())
            trues.append(y.numpy())

    df = pd.DataFrame({
        "path": paths,
        "y_true": np.concatenate(trues),
        "y_pred": np.concatenate(preds),
        "confidence": np.concatenate(confs),
    })
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
