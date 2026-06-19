"""Minimal Grad-CAM for verifying *where* a CNN looks — a key error-analysis
tool (is the model attending to the object, or to background/artifacts?).

Usage (programmatic):
    from src.gradcam import grad_cam
    heat = grad_cam(model, input_tensor, target_layer)   # (H, W) in [0,1]
"""

from __future__ import annotations


def grad_cam(model, input_tensor, target_layer, target_class: int | None = None):
    """Return a normalized (H, W) class-activation heatmap for one image
    (input_tensor shape (1,C,H,W)). target_layer is a conv module of the model."""
    import torch

    activations, gradients = {}, {}

    def fwd_hook(_m, _i, out):
        activations["v"] = out.detach()

    def bwd_hook(_m, _gi, go):
        gradients["v"] = go[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.eval()
        with torch.enable_grad():                       # robust to an enclosing no_grad()
            inp = input_tensor.clone().requires_grad_(True)
            logits = model(inp)
            cls = int(logits.argmax(1)) if target_class is None else target_class
            model.zero_grad()
            logits[0, cls].backward()

        acts = activations["v"][0]            # (K, h, w)
        grads = gradients["v"][0]             # (K, h, w)
        weights = grads.mean(dim=(1, 2))      # (K,)
        cam = torch.relu((weights[:, None, None] * acts).sum(0))
        # upsample to the input resolution so the heatmap is (H, W) as documented
        cam = torch.nn.functional.interpolate(
            cam[None, None], size=inp.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-12)
        return cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
