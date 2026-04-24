"""MONAI DenseNet121-3D backbone.

MONAI ships a ready-to-use 3D DenseNet121 architecture. PyTorch Hub's
pretrained weights are 2D-only (ImageNet), so for 3D spatial_dims we
instantiate randomly-initialised weights and let the task-specific
fine-tune fit them; Med3D (the other backbone) is the right pick when
true 3D pretraining is wanted (trained on 23 public CT datasets).

Reference: monai.networks.nets.DenseNet121
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MonaiDenseNetClassifier(nn.Module):
    def __init__(self, freeze_features: bool = False):
        super().__init__()
        from monai.networks.nets import DenseNet121

        # pretrained=True is rejected for spatial_dims>2 by MONAI — PyTorch
        # Hub only publishes 2D ImageNet weights. Train-from-scratch 3D is
        # the honest default here; freeze_features off because there's no
        # pretrained trunk worth freezing.
        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            pretrained=False,
        )
        if freeze_features:
            # Freeze every layer except the final classifier head. Only
            # meaningful if you've hand-loaded pretrained 3D weights first.
            for name, p in self.model.named_parameters():
                if not name.startswith("class_layers"):
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def load_monai_densenet(freeze_features: bool = False) -> MonaiDenseNetClassifier:
    return MonaiDenseNetClassifier(freeze_features=freeze_features)
