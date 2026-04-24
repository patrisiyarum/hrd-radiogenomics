"""MONAI DenseNet121-3D backbone.

MONAI ships a ready-to-use 3D DenseNet121 with pretrained weights for 2D→3D
inflation. We instantiate the architecture, load MONAI's default ImageNet-
inflated weights, and swap the classifier head for a binary logit.

Reference: monai.networks.nets.DenseNet121
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MonaiDenseNetClassifier(nn.Module):
    def __init__(self, freeze_features: bool = True):
        super().__init__()
        from monai.networks.nets import DenseNet121

        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            pretrained=True,
        )
        if freeze_features:
            # Freeze every layer except the final classifier head.
            for name, p in self.model.named_parameters():
                if not name.startswith("class_layers"):
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def load_monai_densenet(freeze_features: bool = True) -> MonaiDenseNetClassifier:
    return MonaiDenseNetClassifier(freeze_features=freeze_features)
