"""Grad-CAM smoke test on a tiny CNN. Skipped when torch isn't installed (it is
the only torch-dependent test); runs locally / wherever torch is present."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.gradcam import grad_cam  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        feat = self.conv(x)
        return self.fc(self.pool(feat).flatten(1))


def test_gradcam_is_input_sized_and_normalized():
    model = _Tiny()
    x = torch.randn(1, 3, 16, 16)
    cam = grad_cam(model, x, model.conv)
    assert cam.shape == (16, 16)                 # upsampled to input resolution
    assert np.isfinite(cam).all()
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6


def test_gradcam_works_under_no_grad():
    model = _Tiny()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():                        # robust to an enclosing no_grad()
        cam = grad_cam(model, x, model.conv)
    assert cam.shape == (16, 16)
