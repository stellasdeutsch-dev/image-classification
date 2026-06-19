"""Dataset from a folder-per-class layout (ImageFolder style):

    data/dataset/<class_name>/<image>.jpg

Splits into train/val/test and builds transforms. torch/torchvision are imported
lazily so the metrics/error-analysis core stays usable without them."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("imgcls")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_classes(root: str | Path) -> list[str]:
    root = Path(root)
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def build_transforms(image_size: int, train: bool):
    from torchvision import transforms

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])


def build_datasets(cfg: dict):
    """Returns (train_ds, val_ds, test_ds, class_names). Uses a random split
    with a fixed seed; swap for predefined split folders if you have them."""
    import torch
    from torchvision.datasets import ImageFolder

    root = cfg["data_root"]
    size = cfg.get("image_size", 224)
    full_train = ImageFolder(root, transform=build_transforms(size, train=True))
    full_eval = ImageFolder(root, transform=build_transforms(size, train=False))
    class_names = full_train.classes

    n = len(full_train)
    val_frac = cfg.get("val_frac", 0.15)
    test_frac = cfg.get("test_frac", 0.15)
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    n_train = n - n_val - n_test
    g = torch.Generator().manual_seed(cfg.get("seed", 42))
    perm = torch.randperm(n, generator=g).tolist()
    tr_idx, va_idx, te_idx = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]

    train_ds = torch.utils.data.Subset(full_train, tr_idx)
    val_ds = torch.utils.data.Subset(full_eval, va_idx)
    test_ds = torch.utils.data.Subset(full_eval, te_idx)
    log.info("Dataset: %d train / %d val / %d test over %d classes",
             len(train_ds), len(val_ds), len(test_ds), len(class_names))
    return train_ds, val_ds, test_ds, class_names


def class_weights(root: str | Path, class_names: list[str]):
    """Inverse-frequency weights for imbalanced data (-> weighted loss)."""
    import numpy as np

    root = Path(root)
    counts = np.array([sum(1 for p in (root / c).iterdir() if p.suffix.lower() in IMG_EXTS)
                       for c in class_names], dtype=float)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)
    return w
