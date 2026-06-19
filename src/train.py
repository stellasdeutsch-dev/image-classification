"""Training (heavy — run on the GPU via slurm/train.slurm).

Handles class imbalance via inverse-frequency weighted cross-entropy, tracks
val macro-F1, checkpoints the best model, and logs to W&B if enabled.

CLI:  python -m src.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .common import load_config, save_json, seed_everything, select_device, setup_logging

log = logging.getLogger("imgcls")


def run(cfg: dict) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from .data import build_datasets
    from .metrics import confusion_matrix, macro_f1
    from .model import build_model

    seed_everything(cfg.get("seed", 42))
    device = select_device(cfg.get("device", "auto"))
    train_ds, val_ds, _, class_names = build_datasets(cfg)
    save_json(class_names, Path(cfg["out_dir"]) / "classes.json")

    nw = cfg.get("num_workers", 4)
    train_dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True, num_workers=nw)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 32), shuffle=False, num_workers=nw)

    model = build_model(cfg.get("model", "resnet18"), len(class_names), cfg.get("pretrained", True)).to(device)
    weights = None
    if cfg.get("class_weighted_loss", True):
        # inverse-frequency weights from the TRAIN split only (no val/test leak)
        base = train_ds.dataset
        counts = np.bincount([base.samples[i][1] for i in train_ds.indices],
                             minlength=len(class_names)).astype(float)
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (len(counts) * counts)
        weights = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 1e-4))

    run_w = None
    if cfg.get("wandb", False):
        try:
            import wandb
            run_w = wandb.init(project=cfg.get("wandb_project", "image-classification"), config=cfg)
        except Exception as e:  # pragma: no cover
            log.warning("wandb disabled (%s)", e)

    best_f1, best_path = -1.0, Path(cfg["out_dir"]) / "best.pt"
    for epoch in range(cfg.get("epochs", 15)):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * len(x)
        train_loss = running / max(len(train_ds), 1)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_dl:
                logits = model(x.to(device))
                preds.append(logits.argmax(1).cpu().numpy())
                trues.append(y.numpy())
        preds = np.concatenate(preds) if preds else np.array([], int)
        trues = np.concatenate(trues) if trues else np.array([], int)
        cm = confusion_matrix(trues, preds, len(class_names))
        val_f1 = macro_f1(cm)
        log.info("epoch %d  train_loss=%.4f  val_macroF1=%.4f", epoch, train_loss, val_f1)
        if run_w is not None:
            run_w.log({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1})
        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "classes": class_names,
                        "arch": cfg.get("model", "resnet18")}, best_path)

    log.info("Best val macro-F1 = %.4f -> %s", best_f1, best_path)
    if run_w is not None:
        run_w.finish()
    return {"best_macro_f1": best_f1, "checkpoint": str(best_path)}


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Train an image classifier.")
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
