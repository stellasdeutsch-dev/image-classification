"""Model factory via timm (falls back to torchvision resnet18 if timm absent).
The import is separated from the build so an ImportError can never silently
mask a wrong architecture: if timm is missing, only 'resnet18' is supported and
anything else raises (so the arch saved in the checkpoint always matches what
was actually built)."""

from __future__ import annotations

import logging

log = logging.getLogger("imgcls")


def build_model(name: str, num_classes: int, pretrained: bool = True):
    try:
        import timm
    except ImportError:
        timm = None

    if timm is not None:
        log.info("Building timm model %s (num_classes=%d, pretrained=%s)", name, num_classes, pretrained)
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    import torch.nn as nn
    from torchvision import models

    if name != "resnet18":
        raise RuntimeError(
            f"timm is not installed; cannot build '{name}'. Install timm (pip install timm) "
            "or set model='resnet18'."
        )
    log.warning("timm not installed; building torchvision resnet18")
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
