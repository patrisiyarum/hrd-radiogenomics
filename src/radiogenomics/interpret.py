"""Interpretability: 3D Grad-CAM saliency maps.

For each prediction we want to know *why* — specifically, which voxels of
the tumor the model attended to. Grad-CAM produces a coarse heatmap at the
last convolutional layer's resolution; Grad-CAM++ does a weighted
variant that handles multi-object saliency better. Both work on 3D CNNs
with minor adaptations from the `grad-cam` package.

Output is a 3D numpy array the same shape as the preprocessed input
(TARGET_SHAPE = 96^3) with values in [0, 1], ready to overlay on the
original CT in a radiology viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


def grad_cam_3d(
    model: "torch.nn.Module",
    volume: np.ndarray,           # (Z, Y, X), preprocessed to TARGET_SHAPE
    target_class: int = 1,        # 1 = HRD
    target_layer_name: str | None = None,
) -> np.ndarray:
    """Compute a 3D Grad-CAM heatmap aligned with the input volume.

    `target_layer_name` lets you pick which convolutional block to attend
    to. Default: the last block before the classifier head, which tends
    to encode the highest-level concepts.
    """
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    target_layer = _resolve_target_layer(model, target_layer_name)

    # GradCAM expects (B, C, D, H, W); our preprocessed volumes are (Z, Y, X).
    x = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])
    heatmap = grayscale[0]  # (D, H, W)

    # Normalise to [0, 1] so downstream overlays work identically for any
    # model + patient combination.
    heatmap -= heatmap.min()
    denom = heatmap.max() - heatmap.min()
    if denom > 1e-6:
        heatmap /= denom
    return heatmap


def _resolve_target_layer(model, target_layer_name: str | None):
    """Pick a sensible default target layer when the caller doesn't specify."""
    if target_layer_name is None:
        # MONAI DenseNet121's last convolutional block is .features.denseblock4.
        # Med3D ResNet-50's last residual stage is .backbone.layers[3].
        for candidate in (
            "features.denseblock4",
            "backbone.layers.3",
        ):
            module = _get_submodule(model, candidate)
            if module is not None:
                return module
        raise ValueError(
            "couldn't auto-locate a Grad-CAM target layer; pass "
            "target_layer_name explicitly"
        )
    module = _get_submodule(model, target_layer_name)
    if module is None:
        raise ValueError(f"no submodule named {target_layer_name!r}")
    return module


def _get_submodule(model, dotted_name: str):
    current = model
    for part in dotted_name.split("."):
        if part.isdigit():
            current = current[int(part)]
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def save_overlay(
    ct: np.ndarray,
    heatmap: np.ndarray,
    out_path: Path,
    slice_idx: int | None = None,
) -> None:
    """Dump a mid-sagittal slice of (CT + Grad-CAM overlay) to a PNG.

    Used for per-patient interpretability reports in reports/saliency/.
    """
    import matplotlib.pyplot as plt

    if slice_idx is None:
        slice_idx = ct.shape[0] // 2
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(ct[slice_idx], cmap="gray")
    ax.imshow(heatmap[slice_idx], cmap="jet", alpha=0.4)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
