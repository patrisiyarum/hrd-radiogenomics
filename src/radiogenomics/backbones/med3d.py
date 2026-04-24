"""Med3D-architecture ResNet-50 3D backbone.

Chen et al. 2019 released ResNet-10/18/34/50/101/152 3D variants pretrained
on a large medical-imaging corpus (MedNet / 23 public CT datasets). The
canonical weights ship from Tencent/MedicalNet (`resnet_50_23dataset.pth`).

This module constructs the architecture via MONAI's generic 3D ResNet
builder (identical topology: [3, 4, 6, 3] bottleneck blocks, 64 base
filters, 1-channel input, 1-logit output). If a Med3D `.pth` weights
file is present at `models/med3d/resnet_50_23dataset.pth`, we load it
into the backbone; otherwise the model trains from scratch — architecture-
faithful but without the MedNet prior. Fine for a v1 baseline; drop in
the external pretrained weights when you want the full transfer-learning
story.

Weights (optional but recommended):
    https://github.com/Tencent/MedicalNet → resnet_50_23dataset.pth
    → place at models/med3d/resnet_50_23dataset.pth
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path("models/med3d/resnet_50_23dataset.pth")


class Med3DResNet50(nn.Module):
    def __init__(self, freeze_until_block: int = 0):
        super().__init__()
        from monai.networks.nets import ResNet

        # MONAI's ResNet with bottleneck blocks == ResNet-50 topology.
        # spatial_dims=3 gives full 3D convs + 3D pooling; num_classes=1
        # emits a single logit for binary HRD.
        self.backbone = ResNet(
            block="bottleneck",
            layers=[3, 4, 6, 3],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
        )
        self._freeze(freeze_until_block)

    def _freeze(self, up_to_block: int) -> None:
        """Freeze ResNet residual stages 1..up_to_block (inclusive) and the stem.

        ResNet-50 has 4 residual stages (layer1..layer4). `freeze_until=0`
        trains everything; `freeze_until=3` trains only the final stage +
        classifier. Useful only when pretrained weights are loaded — from
        scratch, keep everything trainable.
        """
        if up_to_block <= 0:
            return
        for p in self.backbone.conv1.parameters():
            p.requires_grad = False
        for p in self.backbone.bn1.parameters():
            p.requires_grad = False
        stage_modules = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]
        for i, stage in enumerate(stage_modules, start=1):
            if i <= up_to_block:
                for p in stage.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


def load_med3d(
    weights_path: Path = WEIGHTS_PATH,
    freeze_until: int = 0,
) -> Med3DResNet50:
    """Build a 3D ResNet-50. If Med3D-pretrained weights are on disk, load
    them (strict=False so the Med3D classifier head is dropped and our
    1-logit head stays). Otherwise, return a random-init model.

    freeze_until defaults to 0 (train everything) because the typical v0
    Lambda run has no pretrained weights — freezing random layers is a
    no-op at best and actively harmful at worst.
    """
    model = Med3DResNet50(freeze_until_block=freeze_until)
    if weights_path.exists():
        logger.info("loading Med3D pretrained weights from %s", weights_path)
        state = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        # Med3D state dicts are keyed under module.<layer>; strip the prefix.
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        logger.info(
            "Med3D load: %d missing keys, %d unexpected keys (expected: the "
            "MedNet-era fc layer is dropped)", len(missing), len(unexpected),
        )
    else:
        logger.info(
            "no Med3D weights at %s — training 3D ResNet-50 from scratch. "
            "Download resnet_50_23dataset.pth from Tencent/MedicalNet to "
            "enable full transfer learning.", weights_path,
        )
    return model
